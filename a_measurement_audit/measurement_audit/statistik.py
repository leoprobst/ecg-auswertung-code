from common import *
def moment_est(n,sx,sy,sxx,syy,sxy):
    with np.errstate(divide='ignore',invalid='ignore'):
        col=(sx-sy)**2/(2*n)
        msr=((sxx+2*sxy+syy)/2-(sx+sy)**2/(2*n))/(n-1)
        mse=((sxx+syy-2*sxy)/2-col)/(n-1)
        den=msr+mse+2*(col-mse)/n
        icc=(msr-mse)/den
        sd=np.sqrt(np.maximum(0,(sxx+syy-2*sxy-(sy-sx)**2/n)/(n-1)))
        icc=np.where((n>1)&(den>1e-14),icc,np.nan)
        return np.stack([icc,(sy-sx)/n,sd,sd*1.96],axis=-1)
def pair_stats(x,y,subjects,B=1000):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim==1:x=x[:,None];y=y[:,None]
    ok=np.isfinite(x)&np.isfinite(y)
    with np.errstate(invalid='ignore',divide='ignore'):
        center=np.nanmean(np.where(ok,(x+y)/2,np.nan),axis=0)
        scale=np.sqrt(np.nanmean(np.where(ok,((x-center)**2+(y-center)**2)/2,np.nan),axis=0))
    scale=np.where(np.isfinite(scale)&(scale>0),scale,1.)
    x=np.where(ok,(x-center)/scale,0);y=np.where(ok,(y-center)/scale,0)
    ids,iv=np.unique(subjects,return_inverse=True);u=len(ids);k=x.shape[1]
    moments=[]
    for a in [ok.astype(float),x,y,x*x,y*y,x*y]:
        agg=np.zeros((u,k));np.add.at(agg,iv,a);moments.append(agg)
    point=moment_est(*[a.sum(0) for a in moments]);n=ok.sum(0);ns=(moments[0]>0).sum(0)
    boot=np.empty((B,k,4));rng=np.random.default_rng(47009)
    for lo in range(0,B,20):
        b=min(20,B-lo);w=np.zeros((b,u))
        idx=rng.integers(0,u,(b,u))
        for j in range(b):w[j]=np.bincount(idx[j],minlength=u)
        boot[lo:lo+b]=moment_est(*[w@a for a in moments])
    point[:,1:]*=scale[:,None];boot[:,:,1:]*=scale[None,:,None]
    point[ns<30]=np.nan
    return point,boot,n,ns
def subgroup_frame(p,m):
    cols=['subject_id','study_id','hadm_id','admission_match_ambiguous','machine_metadata_usable_for_primary','machine_report','machine_qrs_duration_ms','machine_cart_id','ecg_time']
    a=m[cols].rename(columns={'study_id':'first_study_id'});a=a.rename(columns={x:x+'_a' for x in cols if x not in ['subject_id','study_id']})
    b=m[cols].rename(columns={'study_id':'second_study_id'});b=b.rename(columns={x:x+'_b' for x in cols if x not in ['subject_id','study_id']})
    z=p.merge(a,on=['subject_id','first_study_id'],how='left',validate='many_to_one').merge(b,on=['subject_id','second_study_id'],how='left',validate='many_to_one')
    def rhythm(s):
        s=s.fillna('').str.lower();return np.where(s.str.contains('atrial fibrillation|atrial flutter'),'AF',np.where(s.str.contains('sinus rhythm'),'Sinus','sonstige'))
    ra,rb=rhythm(z.machine_report_a),rhythm(z.machine_report_b);z['rhythm']=np.where(ra==rb,ra,'wechsel')
    qa,qb=z.machine_qrs_duration_ms_a,z.machine_qrs_duration_ms_b
    z['qrs_class']=np.where(qa.isna()|qb.isna(),'fehlend',np.where((qa>=120)==(qb>=120),np.where(qa>=120,'breit','schmal'),'wechsel'))
    sa=z.machine_report_a.fillna('').str.lower();sb=z.machine_report_b.fillna('').str.lower()
    z['pacing']=np.where(sa.str.contains('paced|pacemaker')&sb.str.contains('paced|pacemaker'),'beide',np.where(sa.str.contains('paced|pacemaker')|sb.str.contains('paced|pacemaker'),'wechsel_oder_einseitig','kein_Report'))
    z['block']=np.where(sa.str.contains('left bundle|\\blbbb\\b')&sb.str.contains('left bundle|\\blbbb\\b'),'LBBB',np.where(sa.str.contains('right bundle|\\brbbb\\b')&sb.str.contains('right bundle|\\brbbb\\b'),'RBBB','kein_beidseitiger_Blockreport'))
    z['cart']=np.where(z.machine_cart_id_a.notna()&z.machine_cart_id_b.notna()&(z.machine_cart_id_a==z.machine_cart_id_b),z.machine_cart_id_a.fillna(-1).astype(int).astype(str),'wechsel_oder_fehlend')
    z['year']=pd.to_datetime(z.first_ecg_time).dt.year.astype(str)
    z['stable']=(z.hadm_id_a.notna()&(z.hadm_id_a==z.hadm_id_b)&z.hours_between.gt(0)&z.hours_between.le(6)&(z.rhythm=='Sinus')&z.qrs_class.isin(['breit','schmal'])&(z.pacing=='kein_Report')&~z.admission_match_ambiguous_a.fillna(True)&~z.admission_match_ambiguous_b.fillna(True)&z.machine_metadata_usable_for_primary_a.fillna(False)&z.machine_metadata_usable_for_primary_b.fillna(False))
    z['stable_quality']=z.stable&z.signal_quality_first.ge(.8)&z.signal_quality_second.ge(.8)
    return z
