import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from .common import *
import pyarrow.dataset as ds
def quantiles(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    return [float(x) for x in np.quantile(v,[.25,.5,.75,.975])] if len(v) else [np.nan]*4
def switch_rows(f,pop,fold,source,marker='spatial_qrs_t_angle_deg',pre='dower__'):
    x=f[('kors__' if 'kors__'+marker in f else '')+marker].to_numpy(float);y=f[pre+marker].to_numpy(float)
    good=np.isfinite(x)&np.isfinite(y);xx=x[good];yy=y[good];ids=f.subject_id.to_numpy()[good]
    uid,iv=np.unique(ids,return_inverse=True);rng=np.random.default_rng(SEED);rows=[]
    w=np.zeros((B,len(uid)))
    for b in range(B):w[b]=np.bincount(rng.integers(0,len(uid),len(uid)),minlength=len(uid))
    for th in [105,90,120]:
        for direction,indicator in [('alle',(xx>th)!=(yy>th)),('Kors_unten_Dower_oben',(xx<=th)&(yy>th)),('Kors_oben_Dower_unten',(xx>th)&(yy<=th))]:
            counts=np.bincount(iv,weights=indicator,minlength=len(uid));den=np.bincount(iv,minlength=len(uid));vals=(w@counts)/(w@den)
            # Personenanteil mindestens ein Methodenwechsel; bei ein EKG/Person identisch.
            anyperson=np.zeros(len(uid));np.maximum.at(anyperson,iv,indicator)
            rows.append({'population':pop,'fold':fold,'marker_id':marker,'threshold':th,'direction':direction,'n_candidate':len(f),'n_pairs':int(good.sum()),'n_subjects':len(uid),'switch_count':int(indicator.sum()),'fraction':float(indicator.mean()),'ci_low':ci(vals)[0],'ci_high':ci(vals)[1],'persons_switch_count':int(anyperson.sum()),'person_fraction':float(anyperson.mean()),'source':source})
    return rows
def clinical():
    inputs=[];rows=[]
    names=['spatial_qrs_t_angle_deg','spatial_peak_qrs_t_angle_deg'];files=sorted((VAR/'bloecke').glob('*.parquet'))
    cols=['subject_id','study_id','split']+[p+n for p in ('kors__','dower__') for n in names]
    f=pd.concat([read(p,cols) for p in files],ignore_index=True);inputs+=files
    for fold in ['train','validation','pool']:
        z=f if fold=='pool' else f[f.split==fold]
        for n in names:rows+=switch_rows(z,'Varianten',fold,str(VAR/'bloecke'),n)
    src=Path(str(data_path("a.mimic_validation_features_primary_core_sentinel_discovery_reserve_classical_criteria_development_split_train")))
    files=sorted(src.glob('part-*.parquet'));selected=files[::max(1,len(files)//60)][:60]
    assert len(selected)==60
    cols=['subject_id','study_id','extraction_status']+[p+n for p in ('','sensitivity_inverse_dower__') for n in names]
    bag=[]
    for p in selected:
        guard('historical_clinical');pf=pq.ParquetFile(p)
        # Peak may be missing from historical schema; area always required.
        keep=[c for c in cols if c in pf.schema_arrow.names]
        bag.append(read(p,keep));inputs.append(p)
    f=pd.concat(bag,ignore_index=True).sort_values(['subject_id','study_id']).drop_duplicates('subject_id');f=f[f.extraction_status=='ok']
    for n in names:
        if n in f and 'sensitivity_inverse_dower__'+n in f:rows+=switch_rows(f,'historisch_60_Trainpartitionen','train',str(src),n,'sensitivity_inverse_dower__')
    pd.DataFrame(rows).to_csv(OUT/'clinical_threshold.csv',index=False)
    bind('clinical',inputs,[OUT/'clinical_threshold.csv'])
def pduration():
    src=MA/'referenz_privat.parquet';f=read(src,['subject_id','study_id','split','age','sex','machine_report','machine_qrs_duration_ms','p_duration_ms'])
    signals=[json.loads(p.read_text()) for p in sorted((MA/'signal_cache').glob('*.json'))]
    s=pd.DataFrame(signals)[['subject_id','study_id','status','p_duration_new']]
    f=f.merge(s,on=['subject_id','study_id'],validate='one_to_one');f['cache']=np.isfinite(f.p_duration_ms);f['fresh']=np.isfinite(f.p_duration_new)
    text=f.machine_report.fillna('').str.lower();f['rhythm']=np.where(text.str.contains('atrial fibrillation|atrial flutter'),'AF_Flutter',np.where(text.str.contains('sinus rhythm'),'Sinus','sonstige'))
    masks={'cache_ja':f.cache,'cache_nein':~f.cache,'cache_ja_fresh_ja':f.cache&f.fresh,'cache_nein_fresh_ja':~f.cache&f.fresh,'cache_nein_fresh_nein':~f.cache&~f.fresh,'fresh_alle':f.fresh,'volle_referenz':np.ones(len(f),bool)}
    # Actual acquisition quality is not present in reference; project only from original whitelist.
    runfeatures=sorted((RUN/'analysis_feature_cache').glob('*/split=train/features.parquet'))+sorted((RUN/'analysis_feature_cache').glob('*/split=validation/features.parquet'))
    quality=[];quality_col=None
    for p in runfeatures:
        pf=pq.ParquetFile(p);c=next((x for x in ['signal_quality_score','signal_quality','quality_score'] if x in pf.schema_arrow.names),None)
        if c:
            quality_col=c;quality.append(ds.dataset(str(p),format='parquet').to_table(columns=['subject_id','study_id',c],filter=ds.field('study_id').isin(f.study_id.tolist())).to_pandas())
    if quality:f=f.merge(pd.concat(quality),on=['subject_id','study_id'],how='left',validate='one_to_one')
    rows=[];dur=[]
    for group,mask in masks.items():
        z=f.loc[mask];row={'group':group,'n':len(z),'age_missing':int(z.age.isna().sum()),'qrs_missing':int(z.machine_qrs_duration_ms.isna().sum()),'signal_failed':int(z.status.ne('ok').sum()),'qrs_ge120_n':int(z.machine_qrs_duration_ms.ge(120).sum())}
        for col in ['age','machine_qrs_duration_ms']+([quality_col] if quality_col else []):
            for label,v in zip(['p25','median','p75','p975'],quantiles(z[col])):row[col+'_'+label]=v
        for col in ['sex','rhythm']:
            for value,cnt in z[col].fillna('fehlend').value_counts().items():row[col+'_'+str(value)+'_n']=int(cnt)
        row['quality_available_column']=quality_col or 'nicht_im_Cache';rows.append(row)
        for col in ['p_duration_ms','p_duration_new']:
            v=z[col].dropna().to_numpy(float);v=v[np.isfinite(v)]
            if not len(v):continue
            rng=np.random.default_rng(SEED);ix=rng.integers(0,len(v),(B,len(v)));bb=v[ix]
            lo,hi=ci(np.quantile(bb,.975,axis=1));ml,mh=ci(np.median(bb,axis=1))
            dur.append({'group':group,'measure':col,'n':len(v),'median':float(np.median(v)),'median_ci_low':ml,'median_ci_high':mh,'p975':float(np.quantile(v,.975)),'p975_ci_low':lo,'p975_ci_high':hi})
    paired=f[f.cache&f.fresh];diff=paired.p_duration_new-paired.p_duration_ms
    bounds=json.loads((MA/'p_ende/fehlerquote.json').read_text());onset=json.loads((MA/'p_ende/grenzquoten.json').read_text())[1]
    summary={'n_all':len(f),'cache_available':int(f.cache.sum()),'cache_missing':int((~f.cache).sum()),'fresh_available':int(f.fresh.sum()),'fresh_only':int((~f.cache&f.fresh).sum()),'paired_duration_n':len(paired),'paired_exact':int(diff.eq(0).sum()),'difference_median':float(diff.median()),'whole_population_confirmed_error_lower':bounds['whole_population_error_lower'],'whole_population_audit_error_upper':bounds['whole_population_error_upper'],'strict_onset_upper':1.,'strict_onset_note':'Kurz/mittel Beginn nicht vergleichbar adjudiziert; 51,36% ist Audit-Gesamtgrenze, keine gesicherte Beginn-Obergrenze.','quality_column':quality_col}
    pd.DataFrame(rows).to_csv(OUT/'p_availability.csv',index=False);pd.DataFrame(dur).to_csv(OUT/'p_duration.csv',index=False);dump(OUT/'p_summary.json',summary)
    f.to_parquet(OUT/'p_characteristics_privat.parquet',index=False)
    bind('p',[src,*sorted((MA/'signal_cache').glob('*.json')),*runfeatures,MA/'p_ende/fehlerquote.json',MA/'p_ende/grenzquoten.json'],[OUT/'p_availability.csv',OUT/'p_duration.csv',OUT/'p_summary.json'])
    print(json.dumps(summary),flush=True)
if __name__=='__main__':clinical();pduration()
