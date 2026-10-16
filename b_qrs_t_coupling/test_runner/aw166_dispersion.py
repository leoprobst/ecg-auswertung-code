"""Descriptive LUDB dispersion by morphology quartiles, no new calibration."""
def run():
    import numpy as np,pyarrow.parquet as pq
    from werkzeuge.b_test.aw166_run import OUT,dump
    p=OUT/'ludb_private.parquet';pf=pq.ParquetFile(p)
    cols=['amplitude_mv','flatness_mv_ms','mean_error_ms','sd_error_ms','n_paired']
    z=pf.read(columns=cols).to_pandas().dropna(subset=cols[:3]);n=len(z)
    x=np.column_stack([np.ones(n),z.amplitude_mv,z.flatness_mv_ms]);y=z.mean_error_ms.to_numpy(float)
    sd=z.sd_error_ms.to_numpy(float);coeff=np.linalg.lstsq(x,y,rcond=None)[0];pred=x@coeff
    cb=np.load(OUT/'calibration_bootstrap_private.npy');n_dev=pq.ParquetFile(OUT/'population_private.parquet').metadata.num_rows
    rng=np.random.default_rng(16620261013);counts=[]
    for b in range(500):
        counts.append(np.bincount(rng.integers(0,n,n),minlength=n))
        rng.integers(0,n_dev,n_dev) # Same reference draws as calibrate().
    counts=np.array(counts);rows=[]
    for measure in ['amplitude_mv','flatness_mv_ms']:
        v=z[measure].to_numpy(float);cuts=np.quantile(v,[0,.25,.5,.75,1])
        for j in range(4):
            group=(v>=cuts[j]) & ((v<cuts[j+1]) if j<3 else (v<=cuts[j+1]))
            res=y[group]-pred[group];within=sd[group];boots=[]
            for b,w in enumerate(counts[:,group]):
                den=w.sum()
                if den<2:continue
                pp=(x@cb[b])[group];rr=y[group]-pp;mean=w@rr/den
                good=np.isfinite(within);wd=w[good].sum()
                boots.append([w@pp/den,np.sqrt(max(0,(w@(rr*rr)-den*mean*mean)/(den-1))),np.sqrt(w[good]@(within[good]**2)/wd) if wd else np.nan])
            arr=np.array(boots)
            rows.append({'section':'LUDB_morphology_dispersion','measure':measure,'quartile':j+1,'lower_inclusive':float(cuts[j]),'upper':float(cuts[j+1]),'upper_inclusive':j==3,'n_records':int(group.sum()),'n_records_with_sd':int(np.isfinite(within).sum()),'n_beats':int(z.loc[group,'n_paired'].sum()),'expected_mean_error_ms':float(pred[group].mean()),'residual_sd_record_mean_ms':float(res.std(ddof=1)),'within_record_rms_sd_ms':float(np.sqrt(np.nanmean(within**2))),'ci95':np.nanquantile(arr,[.025,.975],axis=0).tolist(),'ci_order':['expected_mean_error_ms','residual_sd_record_mean_ms','within_record_rms_sd_ms'],'bootstrap_valid':len(arr),'source':'ludb_private.parquet; fixed LUDB quartiles; aw166_dispersion.py; dispersion.json'})
    dump('dispersion.json',rows);return rows
