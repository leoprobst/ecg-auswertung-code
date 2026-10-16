"""AW054: gebundene Entwicklungsanalyse, additive Vorarbeiten Hauptarbeit B."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import argparse,csv,gc,json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
from . import runner as r,b3
from .resources import guard,Budget,available
OUT=Path(str(data_path("b.b_vorbereitung")))
SPEC=r.PROJECT/'nachrechnung/SPEZIFIKATION_AW054.md'
IDS=['BEF-0063','BEF-0042','BEF-0081','BEF-0049','BEF-0075','BEF-0087']
EXTRA=['classical_cornell_product','classical_sokolow_lyon','magnitude_area_qrs',
 'spatial_low_voltage_qrs_peak_magnitude','classical_right_axis_deviation']
FEATURES=EXTRA+['classical_lbbb_pattern_positive','classical_rbbb_pattern_positive','rhythm_context']
CONTEXT=['preferred_lvef','nearest_echo_delta_hours','ntprobnp_nearest','ntprobnp_nearest_censoring',
 'ntprobnp_reference_upper','ntprobnp_unit','ntprobnp_delta_hours','dx_cardiac_device_status',
 'proc_pacemaker_or_icd','current_report_pacemaker_state','machine_metadata_usable_for_primary','machine_report']
def now():return datetime.now(timezone.utc).isoformat()
def save(p,obj):
 r.OUT=OUT;r.write(OUT/p,obj)
def read(p):return json.loads((OUT/p).read_text(encoding='utf8'))
def check(tag):
 m=guard();OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'ressourcen.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'utc':now(),'block':tag,**m})+'\n')
def bind():
 return {'spec_sha256':r.sha(SPEC),'code_sha256':r.sha(__file__),
 'core_sha256':r.sha(Path(__file__).with_name('core.py')),'resources_sha256':r.sha(Path(__file__).with_name('resources.py'))}
def parquet(p,columns):
 import pyarrow.parquet as pq
 s=pq.ParquetFile(p).schema_arrow
 if set(columns)-set(s.names):raise ValueError('Fehlende Spalten '+str(set(columns)-set(s.names)))
 return pq.read_table(p,columns=columns).to_pandas()
def prepare():
 import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq,pyarrow.dataset as ds,duckdb
 from .core import assert_development
 check('prepare');pa.set_cpu_count(2);pa.set_io_thread_count(2)
 if (OUT/'manifest.json').exists():raise ValueError('Vorbereitung existiert; nicht überschreiben')
 inv=json.loads((b3.OUT/'inventory_AW034.json').read_text(encoding='utf8'))
 items=[];wanted=[];source_hash={str(SPEC):r.sha(SPEC)}
 a=parquet(b3.OUT/'angles_development.parquet',['subject_id','study_id','split',b3.ANGLE])
 am=json.loads((b3.OUT/'angles_manifest.json').read_text(encoding='utf8'))
 assert r.sha(b3.OUT/'angles_development.parquet')==am['projection_file_sha256']
 for part in am['partitions']:
  p=Path(part['path']);assert p.stat().st_size==part['size'] and p.stat().st_mtime_ns==part['mtime_ns']
 for item in inv['items']:
  rp=b3.OUT/item['id']/(item['id']+'.json');result=json.loads(rp.read_text(encoding='utf8'))
  if result['status']!='completed':continue
  assert r.sha(rp)==json.loads(rp.with_suffix('.sha256.json').read_text(encoding='utf8'))['sha256']
  f,old,oldbind=b3.verified_old(item)
  f=f.merge(a,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
  assert r.arrow_sha(pa.Table.from_pandas(f,preserve_index=False))==result['binding']['joined_projection_hash']
  marker=item['key']['feature_id'];cf=old['input_binding']['comparator']['feature_used']
  required=list(dict.fromkeys(['age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',cf,b3.ANGLE,marker]))
  complete=f.dropna(subset=required)
  for split in ('train','validation'):
   z=complete.loc[complete.split==split];flow=result['data']['flow'][split]
   assert len(z)==flow['complete_n'] and int(z.label.sum())==flow['positive']
  pq.write_table(pa.Table.from_pandas(f,preserve_index=False),OUT/(item['id']+'_old.parquet'))
  source_hash[str(rp)]=r.sha(rp)
  item={**item,'comparator':cf,'required':required,'oldbind':oldbind,'b3_result_sha256':r.sha(rp)}
  items.append(item);wanted.append(f[['subject_id','study_id','split']]);del f,complete;gc.collect()
 union=pd.concat(wanted).drop_duplicates().sort_values(['split','study_id'])
 assert not union.study_id.duplicated().any();assert_development(union)
 pq.write_table(pa.Table.from_pandas(union,preserve_index=False),OUT/'universe.parquet')
 rows=[];parts=[]
 for split in ('train','validation'):
  ids=union.loc[union.split==split,'study_id'].tolist()
  for i,p in enumerate(sorted((r.DEV/f'split={split}').glob('part-*.parquet'))):
   if i%25==0:check('feature_'+split+'_'+str(i))
   schema=pq.ParquetFile(p).schema_arrow
   columns=['subject_id','study_id','split']+[c for c in FEATURES if c in schema.names]
   t=ds.dataset(str(p),format='parquet').to_table(columns=columns,filter=ds.field('study_id').isin(ids))
   with p.open('rb') as stream:
    stream.seek(-8,2);n=int.from_bytes(stream.read(4),'little');stream.seek(-8-n,2);footer=stream.read(n+8)
   parts.append({'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
    'footer_sha256':hashlib.sha256(footer).hexdigest(),'projection_sha256':r.arrow_sha(t),'columns':columns})
   if t.num_rows:
    z=t.to_pandas();assert set(z.split)=={split}
    for c in FEATURES:
     if c not in z:z[c]=np.nan
    rows.append(z)
   if i%200==0:print(json.dumps({'phase':'features','split':split,'files':i}),flush=True)
 features=pd.concat(rows,ignore_index=True);assert not features.study_id.duplicated().any()
 features=union.merge(features,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
 for c in EXTRA:features[c]=pd.to_numeric(features[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
 pq.write_table(pa.Table.from_pandas(features,preserve_index=False),OUT/'features.parquet')
 con=duckdb.connect();con.execute("SET memory_limit='1GB'; SET threads=2");con.register('wanted',union)
 context=r.DATA/'evidence/clinical_context_features.parquet';schema=pq.ParquetFile(context).schema_arrow
 assert not set(CONTEXT)-set(schema.names)
 check('context')
 z=con.execute(f"SELECT subject_id,study_id,split,{','.join(CONTEXT)} FROM read_parquet('{context.as_posix()}') WHERE split IN ('train','validation') AND study_id IN (SELECT study_id FROM wanted)").fetchdf()
 assert not z.study_id.duplicated().any() and len(z)==len(union)
 assert_development(z)
 # Bericht ausschließlich als Selektor; Text weder speichern noch loggen.
 report=z.machine_report.fillna('').str.lower()
 z['sinus_selector']=z.machine_metadata_usable_for_primary.fillna(False)&report.str.contains(r'sinus (?:rhythm|bradycardia|tachycardia|arrhythmia)',regex=True)&~report.str.contains(r'atrial fibrillation|atrial flutter|paced|pacemaker',regex=True)
 z=z.drop(columns=['machine_report']);pq.write_table(pa.Table.from_pandas(z,preserve_index=False),OUT/'context.parquet')
 # Attestierte Entwicklungsartefakte vor Zugriff prüfen; keine unbekannten Referenzen zu Kontrollen machen.
 attp=r.DATA/'labels/development_label_attestation.json';att=json.loads(attp.read_text(encoding='utf8'))
 assert att['status']=='verified' and att['scope']=='train_validation_only' and set(att['source_splits'])=={'train','validation'}
 evid=r.DATA/'labels/clinical_label_evidence.parquet';pq.ParquetFile(evid)
 assert r.sha(evid)==att['development_artifacts']['clinical_label_evidence']['sha256']
 check('reference')
 ref=con.execute(f"SELECT subject_id,study_id,split,label_id,label_state,eligibility_state,time_delta_hours FROM read_parquet('{evid.as_posix()}') WHERE split IN ('train','validation') AND label_id IN ('reduced_lvef_strict','elevated_ntprobnp') AND study_id IN (SELECT study_id FROM wanted)").fetchdf()
 assert not ref.duplicated(['study_id','label_id']).any()
 pq.write_table(pa.Table.from_pandas(ref,preserve_index=False),OUT/'references.parquet')
 # Rohverifikationsfluss der endpunktspezifischen Indexpopulation; keine Summierung über Endpunkte.
 lab=r.DATA/'labels/endpoint_index_cohorts.parquet';pq.ParquetFile(lab)
 endpoint_ids=[x['key']['endpoint_id'] for x in items if x['id'] in IDS]
 con.register('endpoint_ids',pd.DataFrame({'label_id':endpoint_ids}))
 counts=con.execute(f"SELECT label_id,split,label_state,eligibility_state,index_selection_strategy,count(*) AS n FROM read_parquet('{lab.as_posix()}') WHERE split IN ('train','validation') AND label_id IN (SELECT label_id FROM endpoint_ids) GROUP BY ALL").fetchdf()
 counts.to_csv(OUT/'reference_flow.csv',sep=';',index=False,encoding='utf-8-sig');con.close()
 save('feature_manifest.json',{'partitions':parts,'sealed_read':False})
 cache=[OUT/'universe.parquet',OUT/'features.parquet',OUT/'context.parquet',OUT/'references.parquet']+[OUT/(x['id']+'_old.parquet') for x in items]
 source_hash.update({str(p):r.sha(p) for p in [attp,evid,r.DATA/'labels/label_definitions.json',b3.OUT/'angles_manifest.json']})
 source_hash[str(context)]={'size':context.stat().st_size,'mtime_ns':context.stat().st_mtime_ns,'selected_projection_sha256':r.sha(OUT/'context.parquet')}
 save('manifest.json',{'created_at':now(),'binding':bind(),'items':items,'sources':source_hash,
   'cache_sha256':{str(p):r.sha(p) for p in cache},'feature_manifest_sha256':r.sha(OUT/'feature_manifest.json'),'sealed_read':False})
 print(json.dumps({'phase':'prepared','items':len(items),'binding':bind()}),flush=True)

def adjust(p,method):
 import numpy as np
 p=np.asarray(p,float);m=len(p);order=np.argsort(p);v=p[order]
 if method=='holm':a=np.maximum.accumulate(v*(m-np.arange(m)))
 elif method=='bh':a=np.minimum.accumulate((v*m/(np.arange(m)+1))[::-1])[::-1]
 else:raise ValueError(method)
 result=np.empty(m);result[order]=np.minimum(1,a);return result

def centered_p(estimate,replicates):
 import numpy as np
 a=np.asarray(replicates,float)
 return float((1+np.count_nonzero(np.abs(a-estimate)>=abs(estimate)))/(len(a)+1))

def resample(universes,subjects,splits,replicate):
 """Gleiche Personenmultiplizitäten unabhängig von Endpunkt/Fallset/Modell."""
 import numpy as np
 rng=np.random.default_rng(np.random.SeedSequence([54009,int(replicate)]));out=np.zeros(len(subjects))
 for split in ('train','validation'):
  u=universes[split];counts=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u));mask=splits==split
  positions=np.searchsorted(u,subjects[mask]);assert np.array_equal(u[positions],subjects[mask])
  out[mask]=counts[positions]
 return out

def fit_raw(x,y,w,warm=None):
 """Gewichtete Z-Skalierung; Vorwärmung in Originaleinheit, identische Ridge."""
 import numpy as np
 from .core import logistic_fit
 mean=np.average(x,axis=0,weights=w);sd=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));sd[sd<1e-12]=1
 z=np.column_stack([np.ones(len(x)),(x-mean)/sd])
 initial=None if warm is None else np.r_[warm[0]+mean@warm[1:],sd*warm[1:]]
 beta=logistic_fit(z,y,w,initial=initial)
 raw=np.r_[beta[0]-mean@(beta[1:]/sd),beta[1:]/sd]
 return raw,z,beta

def sandwich(z,y,beta,subjects):
 import numpy as np
 from scipy.special import expit
 from scipy.stats import norm
 p=expit(z@beta);pen=np.full(z.shape[1],1e-6);pen[0]=0
 bread=z.T@((p*(1-p))[:,None]*z)+np.diag(pen)
 _,inv=np.unique(subjects,return_inverse=True);g=inv.max()+1
 scores=z*(y-p)[:,None];cluster=np.zeros((g,z.shape[1]));np.add.at(cluster,inv,scores)
 n,k=z.shape
 if g<=1 or n<=k:raise ValueError('Sandwich nicht schätzbar')
 inverse=np.linalg.inv(bread);cov=inverse@(cluster.T@cluster)@inverse
 cov*=g/(g-1)*(n-1)/(n-k)
 return cov

def matrices(frame,item,strong=False,nt=False):
 import numpy as np
 from .core import design
 base=np.column_stack([design(frame,'B2',item['comparator']),frame[b3.ANGLE].to_numpy(float)])
 if not strong:return [base]
 axis=np.deg2rad(frame['classical_right_axis_deviation'].to_numpy(float))
 extra=np.column_stack([frame[EXTRA[:-1]].to_numpy(float),np.sin(axis),np.cos(axis)])
 extended=np.column_stack([base,extra])
 if nt:extended=np.column_stack([extended,frame.nt_log_url.to_numpy(float)])
 return [base,extended]

def study(frame,item,universes,name,strong=False,nt=False,iterations=1000,sd_fixed=None):
 import numpy as np
 from scipy.stats import norm
 from .core import auc,interval,pool_summary
 check(item['id']+'_'+name)
 subjects=frame.subject_id.to_numpy();splits=frame.split.to_numpy();y=frame.label.to_numpy(float);marker=frame[item['key']['feature_id']].to_numpy(float)
 masks=[splits=='validation',splits=='train'];ns=[int(m.sum()) for m in masks]
 counts=[{'n':ns[i],'positive':int(y[m].sum()),'negative':int((y[m]==0).sum())} for i,m in enumerate(masks)]
 if any(min(c['positive'],c['negative'])<20 for c in counts):return {'id':item['id'],'scenario':name,'status':'nicht_schaetzbar','reason':'weniger als20 je Klasse/Fold','counts':counts}
 sd=float(np.std(marker)) if sd_fixed is None else float(sd_fixed)
 if sd<1e-12:return {'id':item['id'],'scenario':name,'status':'nicht_schaetzbar','reason':'konstanter Marker','counts':counts}
 arrays=matrices(frame,item,strong,nt);designs=[]
 for x in arrays:
  x=x[:,np.std(x,axis=0)>1e-12]
  full=np.column_stack([x,marker]);standard=(full-full.mean(axis=0))/np.std(full,axis=0)
  if np.linalg.matrix_rank(np.column_stack([np.ones(len(full)),standard]))<full.shape[1]+1:
   return {'id':item['id'],'scenario':name,'status':'nicht_schaetzbar','reason':'Rangdefizit','counts':counts}
  designs.append((x,full))
 warm={}
 def calc(w,point=False):
  if any(np.sum(w[m&(y==v)])<=0 for m in masks for v in (0,1)):raise ValueError('Nur eine Klasse')
  values=[];wald=[]
  for j,(x,full) in enumerate(designs):
   raw,z,beta=fit_raw(full,y,w,warm.get((j,'pool')))
   if point:
    warm[(j,'pool')]=raw
    cov=sandwich(z,y,beta,subjects);rawse=np.sqrt(cov[-1,-1])/np.std(full[:,-1]);se=rawse*sd
    b=float(raw[-1]*sd);wald.append({'beta':b,'se_cluster':float(se),'p_wald':float(2*norm.sf(abs(b/se))),
      'or_wald_ci95':np.exp([b-1.95996398454*se,b+1.95996398454*se]).tolist()})
   held=[]
   for i,m in enumerate(masks):
    fitmask=masks[1-i];scores=[]
    for t,xx in enumerate((x,full)):
     fit,_,_=fit_raw(xx[fitmask],y[fitmask],w[fitmask],warm.get((j,i,t)))
     if point:warm[(j,i,t)]=fit
     scores.append(fit[0]+xx[m]@fit[1:])
    au=[auc(y[m],score,w[m]) for score in scores];held.append(au+[au[1]-au[0]])
   values.extend([float(raw[-1]*sd),*np.asarray(held).ravel().tolist()])
  return np.asarray(values),wald
 point,wald=calc(np.ones(len(frame)),True);weights=np.asarray(ns,float)/sum(ns)
 signature=r.digest({'binding':bind(),'cache':r.sha(OUT/(item['id']+'_old.parquet')),'id':item['id'],'scenario':name,'n':ns,'sd':sd,'frame_hash':r.arrow_sha(__import__('pyarrow').Table.from_pandas(frame,preserve_index=False)),'iterations':iterations,'strong':strong,'nt':nt})
 cp=OUT/'checkpoints'/(item['id']+'_'+name+'.json');values=[];attempt=0;rep_ids=[]
 if cp.exists():
  old=json.loads(cp.read_text(encoding='utf8'));assert old['signature']==signature,'Checkpointbindung'
  values=old['values'];attempt=old['attempt'];rep_ids=old['replicate_ids']
 start=time.perf_counter()
 while len(values)<iterations:
  if len(values)%25==0:check(item['id']+'_'+name+'_'+str(len(values)))
  if attempt>=2*iterations:raise RuntimeError('Replikatlimit')
  replicate=attempt;attempt+=1
  try:val,_=calc(resample(universes,subjects,splits,replicate))
  except ValueError as e:
   if str(e)=='Nur eine Klasse':continue
   raise
  values.append(val.tolist());rep_ids.append(replicate)
  if len(values)%25==0:
   save(Path('checkpoints')/cp.name,{'signature':signature,'values':values,'attempt':attempt,'replicate_ids':rep_ids})
   if len(values)%100==0:print(json.dumps({'id':item['id'],'scenario':name,'replicates':len(values),'seconds':round(time.perf_counter()-start,1)}),flush=True)
 boot=np.asarray(values);models={}
 for j in range(len(designs)):
  k=7*j;beta=float(point[k]);metric={**wald[j],'or':float(np.exp(beta)),'beta_ci95':interval(boot[:,k]),
   'or_ci95':np.exp(interval(boot[:,k])).tolist(),'p_bootstrap_beta':centered_p(beta,boot[:,k])}
  for t,key in enumerate(['baseline_auc','augmented_auc','delta_auc']):
   metric[key]=pool_summary(point[[k+1+t,k+4+t]],boot[:,[k+1+t,k+4+t]],weights)
  delta=metric['delta_auc']['pooled'];draws=boot[:,[k+3,k+6]]@weights
  metric['p_paired_delta_bootstrap']=centered_p(delta,draws)
  se=float(np.std(draws,ddof=1));metric['p_delta_normal_context']=float(2*norm.sf(abs(delta/se))) if se>0 else None
  models[['B3','B4_NT' if nt else 'B4'][j]]=metric
 if len(designs)==2:
  gain=pool_summary(point[[8,11]]-point[[1,4]],boot[:,[8,11]]-boot[:,[1,4]],weights)
 else:gain=None
 result={'id':item['id'],'key':item['key'],'scenario':name,'status':'completed','counts':counts,'marker_sd_development':sd,'models':models,
  'B4_minus_B3_baseline_auc':gain,'iterations':iterations,'attempts':attempt,'replicate_ids':rep_ids,'signature':signature,
  'seconds_this_invocation':time.perf_counter()-start,'sealed_read':False}
 save(Path('results')/(item['id']+'_'+name+'.json'),result);return result

def reference_join(f,refs):
 """Referenzen müssen dieselbe Person, dasselbe EKG und denselben Split treffen."""
 result=f.copy()
 for label,short in [('reduced_lvef_strict','echo'),('elevated_ntprobnp','nt')]:
  q=refs.loc[refs.label_id==label,['subject_id','study_id','split','label_state','eligibility_state','time_delta_hours']].copy()
  q=q.rename(columns={c:short+'_'+c for c in ['label_state','eligibility_state','time_delta_hours']})
  result=result.merge(q,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
 result['both_verified']=(result.echo_label_state.isin(['positive','negative'])&(result.echo_eligibility_state=='eligible')&result.nt_label_state.isin(['positive','negative'])&(result.nt_eligibility_state=='eligible'))
 return result

def run(which):
 import numpy as np,pandas as pd
 from .core import assert_development
 manifest=read('manifest.json');assert manifest['binding']==bind(),'Spezifikation/Code geändert'
 for p,h in manifest['cache_sha256'].items():assert r.sha(p)==h,p
 u=parquet(OUT/'universe.parquet',['subject_id','study_id','split']);universes={s:np.sort(u.loc[u.split==s,'subject_id'].unique()) for s in ('train','validation')}
 feat=parquet(OUT/'features.parquet',['subject_id','study_id','split']+FEATURES)
 context=parquet(OUT/'context.parquet',['subject_id','study_id','split']+[c for c in CONTEXT if c!='machine_report']+['sinus_selector'])
 refs=parquet(OUT/'references.parquet',['subject_id','study_id','split','label_id','label_state','eligibility_state','time_delta_hours'])
 results=[]
 with Budget(OUT) as budget:
  for item in manifest['items']:
   if which=='strong' and item['id'] not in IDS:continue
   check(item['id']);p=OUT/(item['id']+'_old.parquet')
   import pyarrow.parquet as pq
   f=parquet(p,pq.ParquetFile(p).schema_arrow.names);assert_development(f)
   original=f.dropna(subset=item['required']).copy()
   if which=='old':scenarios=[('old_B3',original,False,False,1000,None)]
   else:
    f=f.merge(feat,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
    f=f.merge(context,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
    f=reference_join(f,refs)
    f['nt_log_url']=np.log1p(pd.to_numeric(f.ntprobnp_nearest,errors='coerce')/pd.to_numeric(f.ntprobnp_reference_upper,errors='coerce'))
    # Gebundener Release lab_intervals.py: NULL-Qualifier bedeutet exakten vorhandenen Wert,
    # während below/above_limit ausdrücklich zensiert sind. Fehlender Wert bleibt fehlend.
    nt_exact=f.ntprobnp_nearest_censoring.isna()&f.ntprobnp_reference_upper.gt(0)&f.ntprobnp_nearest.ge(0)&f.ntprobnp_delta_hours.between(-6,12)&f.nt_log_url.notna()
    f.loc[~nt_exact,'nt_log_url']=np.nan
    complete=f.dropna(subset=item['required']+EXTRA).copy();sd=float(complete[item['key']['feature_id']].std(ddof=0))
    echo7=(complete.time_delta_hours.abs().le(168) if item['key']['endpoint_id'] in ('reduced_lvef_strict','echo_rv_systolic_dysfunction') else complete.preferred_lvef.notna()&complete.nearest_echo_delta_hours.abs().le(168))
    clean=complete.qrs_duration_ms.lt(120)&complete.classical_lbbb_pattern_positive.eq(False)&complete.classical_rbbb_pattern_positive.eq(False)&complete.dx_cardiac_device_status.eq(False)&complete.proc_pacemaker_or_icd.eq(False)&complete.current_report_pacemaker_state.eq('negative')
    scenarios=[('strong',complete,True,False,1000,sd),('both_references',complete.loc[complete.both_verified],True,False,1000,sd)]
    scenarios += [(name,complete.loc[mask],True,False,250,sd) for name,mask in [
      ('echo7',echo7),('both_references_echo7',complete.both_verified&echo7),('no_wide_block_pacing',clean),
      ('sinus',complete.sinus_selector.fillna(False)),('age_lt65',complete.age.lt(65)),('age_ge65',complete.age.ge(65)),
      ('sex_F',complete.sex_numeric.eq(0)),('sex_M',complete.sex_numeric.eq(1))]]
    if item['id'] in ['BEF-0063','BEF-0087']:
     scenarios.append(('both_references_NT_numeric',complete.loc[complete.both_verified].dropna(subset=['nt_log_url']),True,True,1000,sd))
    availability=[]
    for split,z in f.groupby('split'):
     for cls,zz in z.groupby('label'):
      availability.append({'split':split,'label':int(cls),'n_index_binary':len(zz),'n_old_b3':int(zz[item['required']].notna().all(axis=1).sum()),
       'n_strong':int(zz[item['required']+EXTRA].notna().all(axis=1).sum()),'n_both_verified':int(zz.both_verified.sum()),
       'n_numeric_lvef':int(zz.preferred_lvef.notna().sum()),'n_numeric_nt':int(zz.ntprobnp_nearest.notna().sum()),'n_nt_exact_url_window':int(zz.nt_log_url.notna().sum()),
       'missing_per_column':{c:int(zz[c].isna().sum()) for c in item['required']+EXTRA},
       'rhythm_context':{str(k):int(v) for k,v in zz.rhythm_context.value_counts(dropna=False).items()},
       'nt_censoring':{str(k):int(v) for k,v in zz.ntprobnp_nearest_censoring.value_counts(dropna=False).items()},
       'nt_units':{str(k):int(v) for k,v in zz.ntprobnp_unit.value_counts(dropna=False).items()}})
    save(Path('availability')/(item['id']+'.json'),{'id':item['id'],'key':item['key'],'rows':availability})
    # Nur private Detailausgabe, keine Patienten-IDs im Projekt.
    complete.to_parquet(OUT/(item['id']+'_strong_private.parquet'),index=False)
   for name,z,strong,nt,iterations,sd in scenarios:
    print(json.dumps({'phase':'start','id':item['id'],'scenario':name}),flush=True)
    rp=OUT/'results'/(item['id']+'_'+name+'.json')
    if rp.exists():
     result=json.loads(rp.read_text(encoding='utf8'));assert result.get('binding',bind())==bind()
    else:
     result=study(z,item,universes,name,strong,nt,iterations,sd)
     result['binding']=bind();save(Path('results')/rp.name,result)
    results.append(result)
   del f,original;gc.collect()
  save('resources_'+which+'.json',{'created_at':now(),**budget.summary(),'binding':bind()})
 print(json.dumps({'phase':'finished','block':which,'results':len(results)}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','old','strong']);args=p.parse_args()
 check('start')
 if args.phase=='prepare':
  with Budget(OUT) as budget:prepare();save('resources_prepare.json',budget.summary())
 else:run(args.phase)
if __name__=='__main__':main()
