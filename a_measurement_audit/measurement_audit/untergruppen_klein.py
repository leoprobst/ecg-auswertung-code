from common import *
import csv,gc
from scipy.stats import spearmanr
dout=OUT/'repeatability';z=pd.read_parquet(dout/'paargruppen_privat.parquet');keys=['subject_id','split','first_study_id','second_study_id'];counts=json.loads((dout/'nenner.json').read_text())
masks={}
for name,cnt in counts.items():
    if cnt['subjects']>=30 or '/' not in name:continue
    pop,dim,val=name.split('/',2);m=(z[dim].astype(str)==val).to_numpy()
    if pop=='stabil':m &= z.stable.to_numpy()
    masks[name]=m
fields=['marker_id','gruppe','n_pairs','n_subjects','bias','difference_sd','rc','sdc95','note']
with (dout/'kleine_strata_deskriptiv.csv').open('w',newline='',encoding='utf-8') as fo:
    wr=csv.DictWriter(fo,fieldnames=fields);wr.writeheader();nrows=0
    for p in sorted((RUN/'repeatability/replicate_pairs').glob('*.parquet')):
        guard('kleine_strata_'+p.stem);names=footer(p);ms=[x[:-6] for x in names if x.endswith('_first') and x[:-6]+'_second' in names and x!='signal_quality_first']
        a=pd.read_parquet(p,columns=keys+[m+s for m in ms for s in ['_first','_second']]).set_index(keys).reindex(pd.MultiIndex.from_frame(z[keys])).reset_index();assert a[keys].equals(z[keys])
        x=a[[m+'_first' for m in ms]].to_numpy(dtype=float,na_value=np.nan);y=a[[m+'_second' for m in ms]].to_numpy(dtype=float,na_value=np.nan);del a
        for group,mask in masks.items():
            for j,m in enumerate(ms):
                good=mask&np.isfinite(x[:,j])&np.isfinite(y[:,j]);d=y[good,j]-x[good,j];sd=float(np.std(d,ddof=1)) if len(d)>1 else np.nan;rc=1.96*sd
                wr.writerow({'marker_id':m,'gruppe':group,'n_pairs':len(d),'n_subjects':z.loc[good,'subject_id'].nunique(),'bias':float(d.mean()) if len(d) else np.nan,'difference_sd':sd,'rc':rc,'sdc95':rc,'note':'fewer than 30 persons; descriptive only, no ICC, confidence interval or suitability decision'});nrows+=1
        del x,y;gc.collect()
print('SMALL_REPEAT',len(masks),nrows,flush=True)
# The exact 60 existing feature partitions, again projected to the 37 marker twins only.
orig=pd.read_parquet(RUN/'guideline/transform_agreement.parquet',columns=['marker_id']);ms=orig.marker_id.tolist();manifest=json.loads((OUT/'transform_inputs.json').read_text())
files=[Path(p) for p in manifest['files'] if '/features/' in p.replace('\\','/') and p.endswith('.parquet')]
cols=['subject_id','study_id','extraction_status']+[x for m in ms for x in [m,'sensitivity_inverse_dower__'+m]]
assert len(files)==60
f=pd.concat([pd.read_parquet(p,columns=cols) for p in files],ignore_index=True).sort_values(['subject_id','study_id']).drop_duplicates('subject_id');f=f[f.extraction_status=='ok']
m=pd.read_parquet(OUT/'master_dev_privat.parquet',columns=['subject_id','study_id','machine_report','machine_qrs_duration_ms','machine_cart_id','ecg_time']);f=f.merge(m,on=['subject_id','study_id'],how='left',validate='one_to_one');del m
s=f.machine_report.fillna('').str.lower();f['rhythm']=np.where(s.str.contains('atrial fibrillation|atrial flutter'),'AF',np.where(s.str.contains('sinus rhythm'),'Sinus','sonstige'));f['qrs_class']=np.where(f.machine_qrs_duration_ms.isna(),'fehlend',np.where(f.machine_qrs_duration_ms>=120,'breit','schmal'));f['pacing']=np.where(s.str.contains('paced|pacemaker'),'Reportpositiv','kein_Report');f['block']=np.where(s.str.contains('left bundle|\\blbbb\\b'),'LBBB',np.where(s.str.contains('right bundle|\\brbbb\\b'),'RBBB','kein_Blockreport'));f['cart']=f.machine_cart_id.fillna(-1).astype(int).astype(str);f['year']=pd.to_datetime(f.ecg_time).dt.year.astype('Int64').astype(str)
rows=[]
for dim in ['rhythm','qrs_class','pacing','block','cart','year']:
    for val,a in f.groupby(dim):
        if len(a)>=30:continue
        for m in ms:
            x=a[m].to_numpy(dtype=float,na_value=np.nan);y=a['sensitivity_inverse_dower__'+m].to_numpy(dtype=float,na_value=np.nan);good=np.isfinite(x)&np.isfinite(y);d=y[good]-x[good];sd=np.std(d,ddof=1) if len(d)>1 else np.nan;bias=d.mean() if len(d) else np.nan
            rows.append({'marker_id':m,'gruppe':dim+'/'+str(val),'n':len(d),'bias':bias,'sd':sd,'loa_low':bias-1.96*sd,'loa_high':bias+1.96*sd,'spearman':spearmanr(x[good],y[good]).statistic if len(d)>1 and np.std(x[good])>0 and np.std(y[good])>0 else np.nan,'note':'fewer than 30 persons; purely descriptive, no confidence interval'})
pd.DataFrame(rows).to_csv(OUT/'transform/kleine_strata_deskriptiv.csv',index=False)
print('SMALL_TRANSFORM',len(rows))
bind([ROOT/'werkzeuge/messaudit/untergruppen_klein.py',dout/'kleine_strata_deskriptiv.csv',OUT/'transform/kleine_strata_deskriptiv.csv'],'kleine_strata')
