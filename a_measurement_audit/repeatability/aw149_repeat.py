"""Independent re-aggregation of the existing replicate pairs, without new pair formation."""
from aw149_common import *
from scipy.stats import rankdata
import gc
KEYS=['spatial_qrs_t_angle_deg','qrs_duration_ms','p_duration_ms','qtc_fridericia_ms','qtc_bazett_ms']
B=1000;KEY=['subject_id','split','first_study_id','second_study_id']
def estimate(n,sx,sy,sxx,syy,sxy):
    with np.errstate(divide='ignore',invalid='ignore'):
        col=(sx-sy)**2/(2*n)
        msr=((sxx+2*sxy+syy)/2-(sx+sy)**2/(2*n))/(n-1)
        mse=((sxx+syy-2*sxy)/2-col)/(n-1)
        den=msr+mse+2*(col-mse)/n
        icc=np.where((n>=2)&(den>0),np.clip((msr-mse)/den,-1,1),np.nan)
        sd=np.sqrt(np.maximum(0,(sxx+syy-2*sxy-(sy-sx)**2/n)/(n-1)))
        return np.stack([icc,(sy-sx)/n,sd,1.96*sd],axis=-1)
def stats(x,y,subject,ids,w):
    ok=np.isfinite(x)&np.isfinite(y)
    with np.errstate(invalid='ignore'):
        center=np.nanmean(np.where(ok,(x+y)/2,np.nan),axis=0)
        scale=np.sqrt(np.nanmean(np.where(ok,((x-center)**2+(y-center)**2)/2,np.nan),axis=0))
    scale=np.where(np.isfinite(scale)&(scale>0),scale,1.)
    xx=np.where(ok,(x-center)/scale,0);yy=np.where(ok,(y-center)/scale,0)
    iv=np.searchsorted(ids,subject);agg=[]
    for v in [ok.astype(float),xx,yy,xx*xx,yy*yy,xx*yy]:
        a=np.zeros((len(ids),x.shape[1]));np.add.at(a,iv,v);agg.append(a)
    ns=(agg[0]>0).sum(0);pp=estimate(*[a.sum(0) for a in agg]);pp[ns<30]=np.nan
    pp[:,1:]*=scale[:,None];bb=np.empty((B,x.shape[1],4))
    for lo in range(0,B,50):
        b=estimate(*[w[lo:lo+50]@a for a in agg])
        eff=(w[lo:lo+50]>0).astype(float)@(agg[0]>0);b[eff<30]=np.nan
        bb[lo:lo+50]=b
    bb[:,:,1:]*=scale[None,:,None]
    return pp,bb,ok.sum(0),ns
def weights(ids,seed):
    rng=np.random.default_rng(seed);w=np.empty((B,len(ids)))
    for b in range(B):w[b]=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids))
    return w
def agreement(x,y):
    good=np.isfinite(x)&np.isfinite(y);x=x[good];y=y[good];n=len(x)
    if n<100 or np.std(x,ddof=1)<=0 or np.std(y,ddof=1)<=0:return {'n_pairs':n,'class':'nicht_schaetzbar'}
    rx=rankdata(x);ry=rankdata(y);rho=np.corrcoef(rx,ry)[0,1]
    qx=np.clip(np.floor(5*(rx-.5)/n),0,4);qy=np.clip(np.floor(5*(ry-.5)/n),0,4)
    switch=np.mean(qx!=qy);width=3.92*np.std(y-x,ddof=1)/np.std(x,ddof=1)
    cl='stabil' if rho>=.95 and width<=.5 and switch<=.10 else 'abhaengig' if rho<.80 or width>2 or switch>.40 else 'bedingt'
    return {'n_pairs':n,'rho':rho,'relative_loa_width':width,'quintile_switch':switch,'class':cl}
def main():
    zpath=MA/'repeatability/paargruppen_privat.parquet';pf=pq.ParquetFile(zpath);z=read(zpath,pf.schema_arrow.names)
    assert len(z)==15693 and z.subject_id.nunique()==10187 and set(z.split)<={'train','validation'}
    master=read(MA/'master_dev_privat.parquet',['subject_id','study_id','split','ecg_time','age','sex'])
    assert set(master.split)<={'train','validation'}
    idx=master.set_index(['subject_id','study_id','split'])
    demo=idx.reindex(pd.MultiIndex.from_frame(z[['subject_id','first_study_id','split']].rename(columns={'first_study_id':'study_id'})))
    # Person-level index: earliest selected first recording, matching the preceding analysis stage.
    order=z.sort_values(['subject_id','first_ecg_time','first_study_id']).drop_duplicates('subject_id').index
    persondemo=pd.DataFrame({'subject_id':z.loc[order,'subject_id'].to_numpy(),'age':demo.iloc[order].age.to_numpy(),'sex':demo.iloc[order].sex.to_numpy()}).set_index('subject_id')
    age=z.subject_id.map(persondemo.age);sex=z.subject_id.map(persondemo.sex)
    del master,idx,demo
    # HR from the explicitly projected train/validation cache, filtered to existing pair keys.
    import pyarrow.dataset as ds
    need=np.unique(np.r_[z.first_study_id,z.second_study_id]);hf=[]
    for fold in ['train','validation']:
        p=next((RUN/'analysis_feature_cache').glob('*/split='+fold+'/features.parquet'));pf=pq.ParquetFile(p);bind(p)
        field=next(n for n in ['context_heart_rate_bpm','heart_rate_bpm','heart_rate'] if n in pf.schema_arrow.names)
        hf.append(ds.dataset(str(p),format='parquet').to_table(columns=['subject_id','study_id',field],filter=ds.field('study_id').isin(need)).to_pandas())
    h=pd.concat(hf).set_index(['subject_id','study_id']);assert h.index.is_unique
    ha=h.reindex(pd.MultiIndex.from_arrays([z.subject_id,z.first_study_id]))[field].to_numpy(float)
    hb=h.reindex(pd.MultiIndex.from_arrays([z.subject_id,z.second_study_id]))[field].to_numpy(float)
    dh=np.abs(hb-ha);del h,hf
    def normal(s,usable):
        s=s.fillna('').str.lower()
        return usable.fillna(False)&s.str.contains('normal ecg|within normal limits')&s.str.contains('sinus rhythm')&~s.str.contains('abnormal|infarct|hypertroph|bundle|fascic|axis deviation|atrial fibrillation|atrial flutter|paced|pacemaker')
    norm=(normal(z.machine_report_a,z.machine_metadata_usable_for_primary_a)&normal(z.machine_report_b,z.machine_metadata_usable_for_primary_b)).to_numpy()
    sinus=z.rhythm.eq('Sinus').to_numpy();gap=z.hours_between.to_numpy()
    masks={'gesamt':np.ones(len(z),bool),'proxy':z.stable.to_numpy(),'zeitgleich':gap==0,
      'sinus_beide':sinus,'sonstiger_rhythmus':~sinus,'hf_lt5':dh<5,'hf_5bis15':(dh>=5)&(dh<=15),'hf_gt15':dh>15,
      'gap_0bis15min':(gap>=0)&(gap<=.25),'gap_15bis60min':(gap>.25)&(gap<=1),'gap_60bis120min':(gap>1)&(gap<=2),
      'header_normal':norm,'sinus_hf_le5':sinus&(dh<=5),'normal_sinus_hf_le5':norm&sinus&(dh<=5),
      'sex_F':sex.eq('F').to_numpy(),'sex_M':sex.eq('M').to_numpy(),'age_lt65':age.lt(65).to_numpy(),'age_ge65':age.ge(65).to_numpy()}
    assert masks['zeitgleich'].sum()==360 and z.loc[masks['zeitgleich'],'subject_id'].nunique()==343
    identity=json.loads(bind(BASE/'rv_a1/waveform_identity_summary.json').read_text());assert identity['decoded_identical']==0 and identity['decode_unresolved']==0
    den={g:{'pairs':int(a.sum()),'subjects':int(z.loc[a,'subject_id'].nunique())} for g,a in masks.items()}
    den['hr_missing']={'pairs':int((~np.isfinite(dh)).sum())};dump(OUT/'denominators.json',den)
    z[KEY].assign(delta_hr=dh,age_group=np.where(age<65,'lt65',np.where(age>=65,'ge65','missing')),sex=sex).to_parquet(OUT/'pair_covariates_privat.parquet',index=False)
    catalog=pd.read_csv(bind(ROOT/'nachrechnung/transform_katalog.csv')).set_index('marker_id')
    parts=sorted((RUN/'repeatability/replicate_pairs').glob('*.parquet'));outputs=[]
    for part in parts:
        target=OUT/(part.stem+'_metrics.csv');calpath=OUT/(part.stem+'_calibration.csv')
        # Inputs bound even on resume; checkpoints tied to code+source hash.
        cols=pq.ParquetFile(part).schema_arrow.names;bind(part)
        marker=[c[:-6] for c in cols if c.endswith('_first') and c[:-6]+'_second' in cols and c!='signal_quality_first']
        checkpoint=OUT/(part.stem+'_checkpoint.json')
        seal={'source':sha(part),'code':sha(__file__),'common':sha(Path(__file__).with_name('aw149_common.py')),'pair':sha(zpath)}
        if checkpoint.exists() and json.loads(checkpoint.read_text())==seal and target.exists() and calpath.exists():
            outputs.extend([target,calpath]);continue
        f=read(part,KEY+[m+s for m in marker for s in ['_first','_second']]);assert not f.duplicated(KEY).any()
        f=f.set_index(KEY).reindex(pd.MultiIndex.from_frame(z[KEY]));x=f[[m+'_first' for m in marker]].to_numpy(float);y=f[[m+'_second' for m in marker]].to_numpy(float);del f
        rows=[];cal=[]
        for g,mask in masks.items():
            # Complete catalog for all, proxy, simultaneous and sex/age groups; key measures for state/gap contrasts.
            js=list(range(len(marker))) if g in ['gesamt','proxy','zeitgleich','sex_F','sex_M','age_lt65','age_ge65'] else [j for j,m in enumerate(marker) if m in KEYS]
            if not js:continue
            ids=np.unique(z.subject_id) if g in ['gesamt','proxy'] else np.unique(z.loc[mask,'subject_id'])
            w=weights(ids,77010 if g in ['gesamt','proxy'] else 149010)
            for lo in range(0,len(js),32):
                use=js[lo:lo+32];xx=x[mask][:,use];yy=y[mask][:,use]
                pp,bb,nn,ns=stats(xx,yy,z.loc[mask,'subject_id'].to_numpy(),ids,w)
                for k,j in enumerate(use):
                    m=marker[j];good=np.isfinite(xx[:,k])&np.isfinite(yy[:,k]);d=yy[good,k]-xx[good,k]
                    row={'population':g,'marker_id':m,'unit':catalog.loc[m,'unit'],'registry_family':catalog.loc[m,'registry_family'],'n_pairs':int(nn[k]),'n_subjects':int(ns[k]),'icc':pp[k,0],'bias':pp[k,1],'difference_sd':pp[k,2],'rc':pp[k,3],'sdc95':pp[k,3], 'source':str(part),'run':RUN.name,'filter':g,'script':'werkzeuge/rv_a1/aw149_repeat.py'}
                    for v,name in enumerate(['icc','bias','difference_sd','rc']):
                        row[name+'_ci_low'],row[name+'_ci_high']=ci(bb[:,k,v]);row[name+'_bootstrap_finite']=int(np.isfinite(bb[:,k,v]).sum())
                    if len(d):
                        row.update(quantile_loa_low=np.quantile(d,.025),quantile_loa_high=np.quantile(d,.975),p95_abs_difference=np.quantile(abs(d),.95),normal_loa_low=pp[k,1]-pp[k,3],normal_loa_high=pp[k,1]+pp[k,3])
                    rows.append(row)
            del w;gc.collect()
        sm=masks['zeitgleich']
        for j,m in enumerate(marker):
            foldclasses=[]
            for fold in ['train','validation','pool']:
                a=sm if fold=='pool' else sm&z.split.eq(fold).to_numpy()
                r=agreement(x[a,j],y[a,j]);r.update(marker_id=m,fold=fold,n_subjects=int(z.loc[a&np.isfinite(x[:,j])&np.isfinite(y[:,j]),'subject_id'].nunique()))
                cal.append(r)
                if fold!='pool':foldclasses.append(r['class'])
            cl='nicht_schaetzbar' if 'nicht_schaetzbar' in foldclasses else 'abhaengig' if 'abhaengig' in foldclasses else 'bedingt' if 'bedingt' in foldclasses else 'stabil'
            cal.append({'marker_id':m,'fold':'worse_fold','class':cl})
        pd.DataFrame(rows).to_csv(target,index=False);pd.DataFrame(cal).to_csv(calpath,index=False);dump(checkpoint,seal)
        outputs.extend([target,calpath]);del x,y;gc.collect();print('COMPLETE',part.name,'rows',len(rows),flush=True)
    frame=pd.concat([pd.read_csv(OUT/(p.stem+'_metrics.csv')) for p in parts],ignore_index=True)
    old=pd.read_csv(bind(BASE/'rv_a1/repeat_marker_metrics.csv'))
    checks=[]
    for g,expected in [('gesamt',.4665168480877628),('proxy',.5242050093521649)]:
        a=frame[frame.population.eq(g)].set_index('marker_id');b=old[old.population.eq(g)].set_index('marker_id').reindex(a.index)
        checks.append({'population':g,'markers_total':len(a),'markers_estimable':int(a.icc.notna().sum()),'median_icc':float(a.icc.median()),'historical_median':expected,'median_delta':float(a.icc.median()-expected),'maximum_icc_delta':float(abs(a.icc-b.icc).max()),'same_estimability':bool(a.icc.notna().equals(b.icc.notna())),'same_n_pairs':bool((a.n_pairs==b.n_pairs).all()),'same_n_subjects':bool((a.n_subjects==b.n_subjects).all())})
    dump(OUT/'core_reproduction.json',checks)
    assert all(c['maximum_icc_delta']<1e-12 and abs(c['median_delta'])<1e-12 and c['same_estimability'] and c['same_n_pairs'] and c['same_n_subjects'] for c in checks)
    frame.to_csv(OUT/'all_metrics.csv',index=False)
    pd.concat([pd.read_csv(OUT/(p.stem+'_calibration.csv')) for p in parts],ignore_index=True).to_csv(OUT/'calibration.csv',index=False)
    finish('repeat',outputs+[OUT/'core_reproduction.json',OUT/'all_metrics.csv',OUT/'calibration.csv',OUT/'denominators.json'])
    print('REPEAT_DONE',checks,flush=True)
if __name__=='__main__':main()
