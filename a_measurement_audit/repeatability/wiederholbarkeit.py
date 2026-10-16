from .common import *
import gc
KEYS=['spatial_qrs_t_angle_deg','qrs_duration_ms','p_duration_ms','qtc_fridericia_ms','qtc_bazett_ms']
def estimate(m):
    n,sx,sy,sxx,syy,sxy=m
    with np.errstate(divide='ignore',invalid='ignore'):
        col=(sx-sy)**2/(2*n);msr=((sxx+2*sxy+syy)/2-(sx+sy)**2/(2*n))/(n-1)
        mse=((sxx+syy-2*sxy)/2-col)/(n-1);den=msr+mse+2*(col-mse)/n
        icc=np.where((n>=2)&(den>0),np.clip((msr-mse)/den,-1,1),np.nan)
        sd=np.sqrt(np.maximum(0,(sxx+syy-2*sxy-(sy-sx)**2/n)/(n-1)))
    return np.stack([icc,(sy-sx)/n,sd,1.96*sd],axis=-1),den
def moments(x,y,subjects,allids,weights=None):
    if x.ndim==1:x=x[:,None];y=y[:,None]
    ok=np.isfinite(x)&np.isfinite(y)
    with np.errstate(invalid='ignore',divide='ignore'):
        center=np.nanmean(np.where(ok,(x+y)/2,np.nan),axis=0)
        scale=np.sqrt(np.nanmean(np.where(ok,((x-center)**2+(y-center)**2)/2,np.nan),axis=0))
    scale=np.where(np.isfinite(scale)&(scale>0),scale,1.)
    xx=np.where(ok,(x-center)/scale,0);yy=np.where(ok,(y-center)/scale,0)
    iv=np.searchsorted(allids,subjects);u=len(allids);k=x.shape[1];agg=[]
    for v in [ok.astype(float),xx,yy,xx*xx,yy*yy,xx*yy]:
        a=np.zeros((u,k));np.add.at(a,iv,v);agg.append(a)
    ns=(agg[0]>0).sum(0);point,den=estimate([a.sum(0) for a in agg]);point[ns<30]=np.nan
    point[:,1:]*=scale[:,None]
    boot=None
    if weights is not None:
        boot=np.empty((B,k,4))
        for lo in range(0,B,50):
            bm=[weights[lo:lo+50]@a for a in agg];b,_=estimate(bm)
            effective=(weights[lo:lo+50]>0).astype(float)@(agg[0]>0)
            b[effective<30]=np.nan;boot[lo:lo+50]=b
        boot[:,:,1:]*=scale[None,:,None]
    # Equal-person averages, not pair-weighted raw variance.
    count=agg[0];ps=np.divide((agg[1]+agg[2])/2,count,out=np.full((u,k),np.nan),where=count>0)
    bsd=np.nanstd(ps,axis=0,ddof=1)*scale
    return point,boot,ok.sum(0),ns,bsd,den,scale
def main():
    identity=json.loads((OUT/'waveform_identity_summary.json').read_text());assert identity['n_pairs']==360
    src=MA/'repeatability/paargruppen_privat.parquet';z=read(src,['subject_id','split','first_study_id','second_study_id','hours_between','first_ecg_time','second_ecg_time','stable'])
    assert set(z.split)<={'train','validation'}
    ids=np.unique(z.subject_id);rng=np.random.default_rng(SEED);w=np.zeros((B,len(ids)))
    for b in range(B):w[b]=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids))
    proxyids=np.unique(z.loc[z.stable,'subject_id'])
    order=z.sort_values(['subject_id','first_ecg_time','second_ecg_time','first_study_id','second_study_id'])
    one=np.zeros(len(z),bool);one[order.drop_duplicates('subject_id').index]=True
    pone=np.zeros(len(z),bool);pone[order[order.stable].drop_duplicates('subject_id').index]=True
    masks={'gesamt':np.ones(len(z),bool),'ohne_zeitgleich':z.hours_between.ne(0).to_numpy(),'proxy':z.stable.to_numpy(),'gesamt_proxy_personen':z.subject_id.isin(proxyids).to_numpy(),'gesamt_ein_paar':one,'proxy_ein_paar':pone}
    dump(OUT/'repeat_denominators.json',{g:{'pairs':int(a.sum()),'subjects':int(z.loc[a,'subject_id'].nunique())} for g,a in masks.items()})
    parts=sorted((RUN/'repeatability/replicate_pairs').glob('*.parquet'));dest=OUT/'repeat';dest.mkdir(exist_ok=True)
    inputs=[src,OUT/'waveform_identity_summary.json'];keys=['subject_id','split','first_study_id','second_study_id']
    for idx,p in enumerate(parts):
        guard('repeat_block_'+str(idx));target=dest/(p.stem+'.npz');inputs.append(p)
        if target.exists():continue
        cols=pq.ParquetFile(p).schema_arrow.names;names=[c[:-6] for c in cols if c.endswith('_first') and c[:-6]+'_second' in cols and c!='signal_quality_first']
        f=read(p,keys+[m+s for m in names for s in ['_first','_second']]);assert not f.duplicated(keys).any()
        f=f.set_index(keys).reindex(pd.MultiIndex.from_frame(z[keys]));x=f[[m+'_first' for m in names]].to_numpy(float);y=f[[m+'_second' for m in names]].to_numpy(float);del f
        records=[];boots=[]
        for g,mask in masks.items():
            doboot=g in ['gesamt','ohne_zeitgleich','proxy','gesamt_proxy_personen']
            pp,bb,nn,ns,bsd,den,scale=moments(x[mask],y[mask],z.loc[mask,'subject_id'].to_numpy(),ids,w if doboot else None)
            boots.append(bb[:,:,0] if bb is not None else np.full((B,len(names)),np.nan))
            for j,n in enumerate(names):
                good=mask&np.isfinite(x[:,j])&np.isfinite(y[:,j]);delta=y[good,j]-x[good,j]
                record={'population':g,'marker_id':n,'n_pairs':int(nn[j]),'n_subjects':int(ns[j]),'icc':pp[j,0],'bias':pp[j,1],'difference_sd':pp[j,2],'rc':pp[j,3],'sdc95':pp[j,3],'sem':pp[j,2]/np.sqrt(2),'between_person_sd':bsd[j],'pooled_measurement_sd':float(np.std(np.r_[x[good,j],y[good,j]],ddof=1)) if good.sum()>1 else np.nan,'rc_lt_between_sd':bool(pp[j,3]<bsd[j]),'source':str(p)}
                if n in KEYS and bb is not None:
                    for k,key in enumerate(['icc','bias','difference_sd','rc']):record[key+'_ci_low'],record[key+'_ci_high']=ci(bb[:,j,k])
                records.append(record)
        np.savez_compressed(target,names=np.asarray(names),boot=np.stack(boots))
        pd.DataFrame(records).to_csv(dest/(p.stem+'.csv'),index=False)
        del x,y,boots;gc.collect();print('REPEAT',idx+1,len(parts),flush=True)
    frame=pd.concat([pd.read_csv(dest/(p.stem+'.csv')) for p in parts],ignore_index=True)
    frame.to_csv(OUT/'repeat_marker_metrics.csv',index=False)
    ns=[];boots=[]
    for p in parts:
        v=np.load(dest/(p.stem+'.npz'));ns.extend(v['names'].tolist());boots.append(v['boot'])
    bb=np.concatenate(boots,axis=2);groups=list(masks);rows=[];pivot=frame.pivot(index='marker_id',columns='population',values='icc').reindex(ns)
    for i,g in enumerate(groups):
        a=pivot[g].to_numpy();valid=np.isfinite(a);b=bb[i]
        row={'population':g,'markers_total':len(a),'markers_estimable':int(valid.sum()),'markers_ge075':int((a>=.75).sum()),'median_icc':float(np.nanmedian(a)),'fraction_ge075':float((a>=.75).sum()/valid.sum())}
        if i<4:
            row['median_ci_low'],row['median_ci_high']=ci(np.nanmedian(b,axis=1));row['fraction_ci_low'],row['fraction_ci_high']=ci((b>=.75).sum(1)/np.isfinite(b).sum(1))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'repeat_summary.csv',index=False)
    contrasts=[]
    for g in ['gesamt','gesamt_proxy_personen']:
        gi=groups.index(g);pi=groups.index('proxy');a=pivot[g].to_numpy();b=pivot.proxy.to_numpy();mask=np.isfinite(a)&np.isfinite(b)
        x=bb[gi,:,mask].T;y=bb[pi,:,mask].T
        for name,point,boot in [('difference_of_medians',np.median(b[mask])-np.median(a[mask]),np.nanmedian(y,axis=1)-np.nanmedian(x,axis=1)),('median_of_marker_differences',np.median(b[mask]-a[mask]),np.nanmedian(y-x,axis=1)),('difference_reliable_fraction',np.mean(b[mask]>=.75)-np.mean(a[mask]>=.75),np.mean(y>=.75,axis=1)-np.mean(x>=.75,axis=1))]:
            lo,hi=ci(boot);contrasts.append({'comparison':'proxy_minus_'+g,'metric':name,'n_same_markers':int(mask.sum()),'point':float(point),'ci_low':lo,'ci_high':hi,'bootstrap_finite':int(np.isfinite(boot).sum())})
    pd.DataFrame(contrasts).to_csv(OUT/'repeat_contrasts.csv',index=False)
    original=pd.read_csv(ROOT/'publikationen/PUB-01_messvaliditaet/T2_repeatability.csv')
    dump(OUT/'repeat_estimability.json',{'historical_T2_rows':len(original),'historical_status_evaluable':int(original.status.eq('evaluable').sum()),'historical_finite_icc':int(np.isfinite(original.icc).sum()),'AW047_finite_scaled':json.loads((MA/'repeatability/aggregate.json').read_text())[0]['markers_estimable'],'new_scaled_finite':rows[0]['markers_estimable'],'original_numerical_gate':'abs(denominator)>1e-12 (release), pairwise n>=30 status gate; AW047 normalized den>1e-14 and finite subjects>=30; AW077 normalized den>0','original_reliability_gate_true':int(original.reliable_for_individual_change.sum()),'new_gate_ge075':rows[0]['markers_ge075']})
    # Heart-rate marker located by schema; do not reconstruct from QTc.
    allcols={p:pq.ParquetFile(p).schema_arrow.names for p in parts}
    hr=next((n for n in ['heart_rate_bpm','heart_rate','median_heart_rate_bpm'] if any(n+'_first' in cols for cols in allcols.values())),None)
    hrrows=[];hrsource=[]
    if not hr:
        import pyarrow.dataset as ds
        need=list(set(z.first_study_id)|set(z.second_study_id));frames=[]
        featurefiles=sorted((RUN/'analysis_feature_cache').glob('*/split=train/features.parquet'))+sorted((RUN/'analysis_feature_cache').glob('*/split=validation/features.parquet'))
        for p in featurefiles:
            guard('HR_projection');cols=pq.ParquetFile(p).schema_arrow.names
            field=next((n for n in ['context_heart_rate_bpm','heart_rate_bpm','heart_rate'] if n in cols),None)
            if field:
                hr=field;hrsource.append(p);frames.append(ds.dataset(str(p),format='parquet').to_table(columns=['subject_id','study_id',field],filter=ds.field('study_id').isin(need)).to_pandas())
        if frames:
            h=pd.concat(frames).set_index(['subject_id','study_id']);a=h.reindex(pd.MultiIndex.from_arrays([z.subject_id,z.first_study_id]))[hr].to_numpy(float);b=h.reindex(pd.MultiIndex.from_arrays([z.subject_id,z.second_study_id]))[hr].to_numpy(float)
            hrframe=pd.DataFrame({hr+'_first':a,hr+'_second':b})
    if hr:
        if hrsource:f=hrframe
        else:
            p=next(p for p,c in allcols.items() if hr+'_first' in c);f=read(p,keys+[hr+'_first',hr+'_second']).set_index(keys).reindex(pd.MultiIndex.from_frame(z[keys]))
        for g,mask in masks.items():
            a=np.abs(f[hr+'_second'].to_numpy(float)-f[hr+'_first'].to_numpy(float))[mask];a=a[np.isfinite(a)]
            hrrows.append({'population':g,'marker_id':hr,'n_candidate':int(mask.sum()),'n_pairs':len(a),'missing_pairs':int(mask.sum())-len(a),**{k:float(v) for k,v in zip(['min','p25','median','p75','p95','max'],np.quantile(a,[0,.25,.5,.75,.95,1]))},**{'fraction_gt'+str(t):float((a>t).mean()) for t in [5,10,20]}})
    pd.DataFrame(hrrows).to_csv(OUT/'hr_delta.csv',index=False);dump(OUT/'hr_availability.json',{'marker':hr,'source':[str(p) for p in hrsource] if hrsource else 'pair blocks' if hr else 'not found'})
    inputs+=hrsource
    bind('repeat',inputs+[ROOT/'publikationen/PUB-01_messvaliditaet/T2_repeatability.csv'],[OUT/'repeat_marker_metrics.csv',OUT/'repeat_summary.csv',OUT/'repeat_contrasts.csv',OUT/'repeat_estimability.json',OUT/'hr_delta.csv'])
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
