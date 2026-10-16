from common import *
cache=OUT/'signal_cache';data=[json.loads(p.read_text()) for p in sorted(cache.glob('*.json'))]
ref=pd.read_parquet(OUT/'referenz_privat.parquet');assert len(data)==len(ref)==1500
df=pd.DataFrame(data);rows=[]
for agegroup,mask in [('alle',np.ones(len(df),bool)),('20_59',df.age.between(20,59).to_numpy())]:
    for seg in ['qrs','t']:
        for plane in ['frontal','transversal','rechts_sagittal']:
            vals=[r[seg+'_'+plane] for r in data if (agegroup=='alle' or 20<=r['age']<=59) and r.get('status')=='ok' and r.get(seg+'_'+plane) is not None]
            n=len(vals);minor=np.array([v['minor'] for v in vals]);major=np.array([v['majority'] for v in vals]);rng=np.random.default_rng(47009);idx=rng.integers(0,n,(1000,n))
            for threshold in [.05,.10,.15,.25,None]:
                labels=np.where(minor>=threshold,'figure8',major) if threshold is not None else major
                for label in ['cw','figure8','ccw','tie']:
                    indicators=labels==label;rows.append({'pruefung':2,'agegroup':agegroup,'segment':seg,'plane':plane,'threshold':threshold if threshold is not None else 'majority','class':label,'n_candidates':int(mask.sum()),'n_available':n,'count':int(indicators.sum()),'fraction':float(indicators.mean()),'ci':ci(indicators[idx].mean(1)),'source':str(cache),'filter':'status_ok/segment/plane/agegroup','script':'werkzeuge/messaudit/referenz_auswerten.py'})
    # Additional direction concordance with a fixed nonlinearity gate.
    selected=[r for r in data if (agegroup=='alle' or 20<=r['age']<=59) and r.get('status')=='ok' and all(r.get(s+'_transversal') and r[s+'_transversal']['width_ratio']>=.05 and r[s+'_transversal']['majority']!='tie' for s in ['qrs','t'])]
    agree=np.array([r['qrs_transversal']['majority']==r['t_transversal']['majority'] for r in selected]);n=len(agree);rng=np.random.default_rng(47009);idx=rng.integers(0,n,(1000,n))
    rows.append({'pruefung':'2_LKF2','agegroup':agegroup,'n_candidates':int(mask.sum()),'n_available':n,'fraction_same_direction':float(agree.mean()),'ci':ci(agree[idx].mean(1)),'width_gate':.05})
pstats=[]
for agegroup,mask in [('alle',np.ones(len(df),bool)),('18_59',df.age.between(18,59).to_numpy()),('60plus',(df.age>=60).to_numpy())]:
    for column in ['p_duration_cache','p_duration_new']:
        v=pd.to_numeric(df.loc[mask,column],errors='coerce').dropna().to_numpy();n=len(v);rng=np.random.default_rng(47009);idx=rng.integers(0,n,(1000,n))
        pstats.append({'agegroup':agegroup,'measure':column,'n_candidates':int(mask.sum()),'n':n,'median':float(np.median(v)),'p97_5':float(np.quantile(v,.975)),'fraction_gt120':float((v>120).mean()),'fraction_gt130':float((v>130).mean()),'median_ci':ci(np.median(v[idx],axis=1)),'p97_5_ci':ci(np.quantile(v[idx],.975,axis=1)),'gt120_ci':ci((v[idx]>120).mean(1)),'gt130_ci':ci((v[idx]>130).mean(1))})
pd.DataFrame(rows).to_csv(OUT/'achter_aggregate.csv',index=False);jsave(OUT/'achter_aggregate.json',rows);jsave(OUT/'p_ende/dauer_aggregate.json',pstats)
good=np.isfinite(pd.to_numeric(df.p_duration_cache,errors='coerce'))&np.isfinite(pd.to_numeric(df.p_duration_new,errors='coerce'))
jsave(OUT/'signal_aggregate.json',{'n':len(df),'status':df.status.value_counts().to_dict(),'median_status':df.median_status.value_counts().to_dict(),'cache_raw_duration_pairs':int(good.sum()),'cache_raw_duration_equal':int((abs(df.loc[good,'p_duration_cache']-df.loc[good,'p_duration_new'])<1e-8).sum()),'note':'p_duration_new from raw representative fiducials; p_duration_cache from marker gates; not equated with a norm','source':str(cache)})
bind([OUT/'referenz_privat.parquet',OUT/'signale_inputs.json',ROOT/'werkzeuge/messaudit/referenz_auswerten.py',OUT/'achter_aggregate.csv'],'referenz')
print('REFERENCE',json.loads((OUT/'signal_aggregate.json').read_text()))
print('P',json.dumps(pstats));print('QRS15',[r for r in rows if r.get('segment')=='qrs' and r.get('threshold')==.15])
