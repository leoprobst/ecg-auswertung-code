"""Median expert reference on the same QRS-anchored LUDB pairs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
def run():
    from pathlib import Path
    import numpy as np,pandas as pd,pyarrow.parquet as pq,wfdb,statsmodels.api as sm
    from werkzeuge.b_test.aw166_run import OUT,dump,sha
    data=Path(str(data_path("b.lobachevsky_university_electrocardiography_database_1_0_1")))
    base=Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_publication_preparation_research_beat_boundaries_20261012_v1")))
    records=[r.split('/')[-1] for r in (data/'RECORDS').read_text().splitlines() if r]
    leads=['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6'];rows=[];sources={}
    for i,record in enumerate(records,1):
        waves={'N':[],'t':[]}
        for lead in leads:
            p=data/'data'/f'{record}.{lead}';sources[str(p)]=sha(p);ann=wfdb.rdann(str(data/'data'/record),lead)
            for j,symbol in enumerate(ann.symbol):
                if symbol in waves and 0<j<len(ann.symbol)-1 and ann.symbol[j-1]=='(' and ann.symbol[j+1]==')' and ann.sample[j-1]<ann.sample[j]<ann.sample[j+1]:
                    waves[symbol].append({'lead':lead,'peak':int(ann.sample[j]),'end':int(ann.sample[j+1])})
        groups={}
        for kind,items in waves.items():
            cluster=[]
            for item in sorted(items,key=lambda v:(v['peak'],v['lead'])):
                if cluster and item['peak']-cluster[-1][0]['peak']<=40 and item['lead'] not in [v['lead'] for v in cluster[-1]]:cluster[-1].append(item)
                else:cluster.append([item])
            groups[kind]=[{'peak':float(np.median([v['peak'] for v in g])),'end':max(v['end'] for v in g),'median_end':float(np.median([v['end'] for v in g]))} for g in cluster if len(g)>=3]
        p=base/f'errors-{i:03d}.parquet';pf=pq.ParquetFile(p);sources[str(p)]=sha(p)
        z=pf.read(columns=['scope','landmark','reference_beat','reference_sample','prediction_sample','error_ms']).to_pandas()
        z=z[z.scope.eq('fused_global')&z.landmark.eq('t_offset')];values=[]
        assert np.allclose(z.error_ms,2*(z.prediction_sample-z.reference_sample),equal_nan=True)
        for row in z.itertuples():
            q=groups['N'][int(row.reference_beat)];candidates=[t for t in groups['t'] if 0<t['peak']-q['peak']<350]
            t=min(candidates,key=lambda v:v['peak']);assert row.reference_sample==t['end']
            if np.isfinite(row.prediction_sample):values.append(2*(row.prediction_sample-t['median_end']))
        rows.append({'record':record,'median_mean_error_ms':np.mean(values) if values else np.nan,'n_paired':len(values)})
    pq.ParquetFile(OUT/'ludb_private.parquet')
    z=pd.read_parquet(OUT/'ludb_private.parquet',columns=['record','amplitude_mv','flatness_mv_ms']).merge(pd.DataFrame(rows),on='record',validate='one_to_one').dropna(subset=['amplitude_mv','flatness_mv_ms','median_mean_error_ms'])
    x=np.column_stack([np.ones(len(z)),z.amplitude_mv,z.flatness_mv_ms]);y=z.median_mean_error_ms.to_numpy(float)
    model=sm.OLS(y,x).fit(cov_type='HC3')
    pf=pq.ParquetFile(OUT/'population_private.parquet');f=pf.read(columns=['amplitude_mv','flatness_mv_ms','y']).to_pandas()
    xx=np.column_stack([np.ones(len(f)),f.amplitude_mv,f.flatness_mv_ms]);yy=f.y.to_numpy(int);pred=xx@model.params
    rng=np.random.default_rng(np.random.SeedSequence([16620261013,3]));diff=[]
    for b in range(500):
        ix=rng.integers(0,len(z),len(z));beta=np.linalg.lstsq(x[ix],y[ix],rcond=None)[0]
        ix=rng.integers(0,len(f),len(f));pp=xx[ix]@beta;yb=yy[ix];diff.append(pp[yb==1].mean()-pp[yb==0].mean())
    result={'section':'median_reference_calibration','n_records':len(z),'n_beats':int(z.n_paired.sum()),'params':model.params.tolist(),'ci95_robust_HC3':model.conf_int().tolist(),'residual_sd_record_ms':float(model.resid.std(ddof=3)),'mean_shift_low':float(pred[yy==1].mean()),'mean_shift_high':float(pred[yy==0].mean()),'group_difference_ms':float(pred[yy==1].mean()-pred[yy==0].mean()),'group_difference_ci95_ms':np.quantile(diff,[.025,.975]).tolist(),'bootstrap':500,'source':'public LUDB annotations + same AW140 QRS-anchored rows; aw166_reference.py; median_reference.json','scope':'reference/calibration sensitivity only; primary MIMIC signal correction uses extreme reference'}
    dump('median_reference.json',result);dump('median_reference_sources_private.json',sources);return result
