from statistik import *
import gc
dout=OUT/'repeatability';dout.mkdir(exist_ok=True)
parts=sorted((RUN/'repeatability/replicate_pairs').glob('*.parquet'))
basecols=['subject_id','split','first_study_id','second_study_id','first_ecg_time','second_ecg_time','hours_between','intervening_intervention_checked','signal_quality_first','signal_quality_second']
base=pd.read_parquet(parts[0],columns=basecols);assert set(base.split)<={'train','validation'}
master=pd.read_parquet(OUT/'master_dev_privat.parquet');z=subgroup_frame(base,master);del master
z.to_parquet(dout/'paargruppen_privat.parquet',index=False)
groups={'gesamt':np.ones(len(z),bool),'stabil':z.stable.to_numpy(),'stabil_qualitaet':z.stable_quality.to_numpy()}
order=z.sort_values(['subject_id','first_ecg_time','second_ecg_time','first_study_id','second_study_id'])
for pop in ['gesamt','stabil']:
    keep=order if pop=='gesamt' else order[order.stable]
    firstidx=keep.drop_duplicates('subject_id').index
    mask=np.zeros(len(z),bool);mask[firstidx]=True;groups[pop+'_ein_Paar']=mask
    for dim in ['rhythm','qrs_class','pacing','block','cart','year']:
        for val in sorted(z[dim].unique()):
            mask=(z[dim]==val).to_numpy() & (np.ones(len(z),bool) if pop=='gesamt' else z.stable.to_numpy())
            groups[pop+'/'+dim+'/'+str(val)]=mask
jsave(dout/'nenner.json',{g:{'pairs':int(a.sum()),'subjects':int(z.loc[a,'subject_id'].nunique())} for g,a in groups.items()})
aggregates={g:{'icc':[],'boot_icc':[]} for g in groups if '/' not in g}
rows=[];Bmarkers=set(x['feature_id'] for x in json.loads((OUT/'befund_schluessel.json').read_text()));agreement_audit=[]
joined=pd.read_parquet(OUT/'paare_ergaenzt_privat.parquet')
join_keys=['subject_id','split','first_study_id','second_study_id']
joined=joined.set_index(join_keys).reindex(pd.MultiIndex.from_frame(base[join_keys])).reset_index()
for block,p in enumerate(parts):
    rows=[]
    guard('repeat_block_'+str(block));cols=footer(p);markers=[x[:-6] for x in cols if x.endswith('_first') and x[:-6]+'_second' in cols and x not in ['signal_quality_first']]
    a=pd.read_parquet(p,columns=basecols+[x+s for x in markers for s in ['_first','_second']])
    pairkeys=['subject_id','split','first_study_id','second_study_id']
    assert not a.duplicated(pairkeys).any()
    a=a.set_index(pairkeys).reindex(pd.MultiIndex.from_frame(base[pairkeys])).reset_index()
    assert a[basecols].equals(base), 'Spaltenblock-Paaridentitaet verletzt'
    for marker in Bmarkers.intersection(markers):
        for suffix in ['_first','_second']:
            v=a[marker+suffix].to_numpy();w=joined[marker+suffix].to_numpy();agree=np.isclose(v,w,rtol=1e-8,atol=0,equal_nan=True)
            agreement_audit.append({'marker':marker,'measurement':suffix,'n':len(v),'disagree':int((~agree).sum())})
    x=a[[m+'_first' for m in markers]].to_numpy(dtype=float,na_value=np.nan);y=a[[m+'_second' for m in markers]].to_numpy(dtype=float,na_value=np.nan);del a
    completed=(dout/(p.stem+'_kennzahlen.csv')).exists()
    for g,mask in groups.items():
        if completed and g not in aggregates:continue
        ns0=z.loc[mask,'subject_id'].nunique()
        if ns0<30:continue
        point,boot,n,ns=pair_stats(x[mask],y[mask],z.loc[mask,'subject_id'].to_numpy())
        if g in aggregates:aggregates[g]['icc'].extend(point[:,0]);aggregates[g]['boot_icc'].append(boot[:,:,0])
        for j,m in enumerate(markers):
            good=np.isfinite(x[mask,j])&np.isfinite(y[mask,j]);d=y[mask,j][good]-x[mask,j][good]
            row={'marker_id':m,'gruppe':g,'n_pairs':int(n[j]),'n_subjects':int(ns[j]),'icc':point[j,0],'bias':point[j,1],'difference_sd':point[j,2],'rc':point[j,3],'sdc95':point[j,3],'sem':point[j,2]/np.sqrt(2),'individuell_icc_ge075':bool(point[j,0]>=.75),'formal_estimable':bool(np.isfinite(point[j,0])),'median_difference':float(np.median(d)) if len(d) else np.nan,'robust_p95_centered':float(np.quantile(abs(d-np.median(d)),.95)) if len(d) else np.nan,'source':str(p),'script':'werkzeuge/messaudit/wiederholbarkeit.py','filter':g}
            for jstat,name in enumerate(['icc','bias','sd','rc']):row[name+'_ci_low'],row[name+'_ci_high']=ci(boot[:,j,jstat]) if ns[j]>=30 else [np.nan,np.nan]
            rows.append(row)
    if not completed:pd.DataFrame([r for r in rows if r['marker_id'] in markers]).to_csv(dout/(p.stem+'_kennzahlen.csv'),index=False)
    del x,y,point,boot;gc.collect();print('REPEAT_BLOCK',block+1,'/11','rows',len(rows),'RSS',peak,flush=True)
summary=[]
for g,agg in aggregates.items():
    v=np.array(agg['icc']);bb=np.concatenate(agg['boot_icc'],axis=1);estim=np.isfinite(v)
    bvalid=np.isfinite(bb);cnt=bvalid.sum(1);prop=np.divide((bb>=.75).sum(1),cnt,out=np.full(1000,np.nan),where=cnt>0)
    summary.append({'gruppe':g,'markers_total':len(v),'markers_estimable':int(estim.sum()),'markers_reliable':int((v>=.75).sum()),'icc_median':float(np.nanmedian(v)),'icc_median_ci':ci(np.nanmedian(bb,axis=1)),'reliable_fraction_estimable':float((v>=.75).sum()/estim.sum()),'fraction_ci':ci(prop),'reliable_fraction_all':float((v>=.75).mean()),'fraction_all_ci':ci((bb>=.75).mean(1))})
jsave(dout/'aggregate.json',summary);jsave(dout/'cache_gleichheit.json',agreement_audit)
c=conn();c.execute(f"COPY (SELECT * FROM read_csv_auto('{sqlp(dout/'part-*_kennzahlen.csv')}',union_by_name=true)) TO '{sqlp(dout/'alle_kennzahlen.csv')}' (HEADER, DELIMITER ',')");c.close()
bind([*parts,OUT/'master_dev_privat.parquet',ROOT/'werkzeuge/messaudit/statistik.py',ROOT/'werkzeuge/messaudit/wiederholbarkeit.py',dout/'alle_kennzahlen.csv'],'repeatability')
print(json.dumps(summary,ensure_ascii=True))
