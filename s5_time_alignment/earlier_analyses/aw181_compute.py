"""Bounded development-partition computations; no historical generator is executed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os,sys,time,json,hashlib,ast,threading,traceback,re,csv
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
ROOT=Path(__file__).resolve().parents[2];OUT=Path(str(data_path("s5.aw181")))
import psutil
def guard():
 if min(psutil.virtual_memory().available,psutil.swap_memory().free+psutil.virtual_memory().available)<2*2**30:raise MemoryError('2 GiB reserve undershot')
 if psutil.Process().memory_info().rss>1.4*2**30:raise MemoryError('Memory limit reached')
guard()
def watch():
 while True:
  try:guard()
  except MemoryError:
   (OUT/'resource_stop.json').write_text(json.dumps({'pid':os.getpid(),'reason':'Resource gate closed'}));os._exit(75)
  time.sleep(1)
threading.Thread(target=watch,daemon=True).start()
import numpy as np,pandas as pd,duckdb,pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching,maximum_flow
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read(p):
 assert Path(p).stat().st_size<20_000_000
 return json.loads(Path(p).read_text(encoding='utf-8-sig'))
PROTOCOL=sha(OUT/'protocol.json');assert PROTOCOL==(OUT/'protocol.sha256').read_text().strip()
MANIFEST=read(OUT/'input_manifest.json') if (OUT/'input_manifest.json').exists() else {}; SQL=[];RESULT={}
def bind(p):
 p=Path(p);MANIFEST[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size};save('input_manifest.json',MANIFEST);return p
def connection(db=None):
 c=duckdb.connect(str(db),read_only=True) if db else duckdb.connect()
 c.execute("SET memory_limit='512MB'; SET threads=2")
 if not db:c.execute(f"SET temp_directory='{OUT.as_posix()}/spill'")
 return c
def query(c,s,args=None):
 SQL.append(s);return c.execute(s,args or [])
def pure(path,names,ns):
 code=bind(path).read_text(encoding='utf-8-sig');nodes=[n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pure archived functions>','exec'),ns)

def offset():
 db=bind(str(data_path("s5.uhrzeit_aw110_audit_duckdb")));c=connection(db)
 assert query(c,"SELECT count(*) FROM whitelist WHERE fold NOT IN ('train','validation')").fetchone()[0]==0
 assert query(c,'SELECT count(*) FROM (SELECT subject_id FROM whitelist GROUP BY subject_id HAVING count(DISTINCT fold)>1)').fetchone()[0]==0
 D=np.arange(-720,721,5); ns={'np':np,'D':D,'SHIFT':D*60}
 pure(ROOT/'werkzeuge/uhrzeit_aw110/rechnen.py',{'curves','charts','estimate'},ns)
 historical=read(bind(str(data_path("s5.uhrzeit_aw110_icu_results_json"))))
 candidates=query(c,'SELECT subject_id,fold FROM eligible GROUP BY ALL HAVING count(*)>=3 ORDER BY subject_id').fetchall()
 shifts=read(OUT/'protocol.json')['offset_shifts_minutes'];rng=np.random.default_rng(181001)
 base=[]; shifted=[];perm=[];byfold={'train':[],'validation':[]}
 for ix,(person,fold) in enumerate(candidates):
  e=query(c,'SELECT study_id,icustay_id,epoch(ecg_time),hf,anchor_year_group,first_careunit,machine_cart_id,epoch(intime),epoch(outtime) FROM eligible WHERE subject_id=? ORDER BY study_id',[person]).fetchall()
  h=query(c,'SELECT icustay_id,epoch(t),hf FROM hf WHERE subject_id=? ORDER BY icustay_id,t',[person]).fetchall()
  charts=ns['charts'](h);L,meta,_,_=ns['curves'](e,charts)
  if len(L)<3:continue
  loss,off,width,ok,minimum=ns['estimate'](L); near=D[loss<=minimum+1];opts=D[np.isclose(loss,minimum,rtol=0,atol=1e-10)]
  # HF variability among ECGs with complete curves, identified by the unchanged coverage rule.
  complete_e=[z for z in e if len(ns['curves']([z],charts)[0])]
  values=np.array([z[3] for z in complete_e]);zone=[int(near[0]),int(near[-1])]
  rec={'fold':fold,'n':len(L),'offset':off,'width':width,'ok':ok,'minimum':minimum,'argmin_n':len(opts),'tie_mid':float((opts[0]+opts[-1])/2),'near_mid':float(sum(zone)/2),'near_low':zone[0],'near_high':zone[1],'hf_sd':float(np.std(values))}
  base.append(rec);byfold[fold].append(rec)
  for shift in shifts:
   es=[(z[0],z[1],z[2]+shift*60,*z[3:]) for z in e]
   LL,_,_,_=ns['curves'](es,charts)
   if LL:
    ls,o,w,good,mn=ns['estimate'](LL);nn=D[ls<=mn+1]
    expected=off-shift
    shifted.append({'fold':fold,'shift':shift,'n_complete':len(LL),'baseline_ok':ok,'ok':good,'error':int(o-expected),'expected':int(expected),'zone_contains_expected':bool(nn[0]<=expected<=nn[-1])})
   else:shifted.append({'fold':fold,'shift':shift,'n_complete':0,'baseline_ok':ok,'ok':False,'error':None,'expected':int(off-shift),'zone_contains_expected':False})
  for repetition in range(20):
   # Shuffle physiological values across ECG times inside the same person, leave reference unchanged.
   vals=rng.permutation([z[3] for z in e]);ep=[(*z[:3],float(v),*z[4:]) for z,v in zip(e,vals)]
   LP,_,_,_=ns['curves'](ep,charts)
   if LP:
    _,o,w,good,mn=ns['estimate'](LP);perm.append({'fold':fold,'repetition':repetition,'ok':good,'offset':o,'n':len(LP)})
  if ix%300==0:print('Offset persons',ix,'of',len(candidates),flush=True)
  if ix%50==0:guard()
 save('offset_private_metrics.json',{'baseline':base,'shifted':shifted,'permutations':perm})
 summaries=[];sensitivity=[];strata=[];null=[]
 for fold in ['train','validation']:
  b=byfold[fold];good=[x for x in b if x['ok']]
  assert len(b)==historical['folds'][fold]['at_least_3_people'] and len(good)==historical['folds'][fold]['identifiable_people']
  assert sum(abs(x['offset'])<=60 for x in good)==historical['folds'][fold]['within']['60']['count']
  for shift in shifts:
   sub=[x for x in shifted if x['fold']==fold and x['shift']==shift];both=[x for x in sub if x['baseline_ok'] and x['ok']];err=np.array([x['error'] for x in both])
   summaries.append({'fold':fold,'shift_minutes':shift,'baseline_at_least3':len(b),'shifted_at_least3':sum(x['n_complete']>=3 for x in sub),'shifted_identifiable':sum(x['ok'] for x in sub),'jointly_identifiable':len(both),'bias_minutes':float(np.mean(err)) if len(err) else None,'median_error_minutes':float(np.median(err)) if len(err) else None,'median_absolute_error_minutes':float(np.median(abs(err))) if len(err) else None,'within15':int(sum(abs(err)<=15)),'within60':int(sum(abs(err)<=60)),'zone_covers_expected':sum(x['zone_contains_expected'] for x in both),'success':bool(len(err)>0 and np.median(abs(err))<=15 and np.mean(abs(err)<=60)>=.9)})
  for rule in ['offset','tie_mid','tie_exclude','near_low','near_high','near_mid']:
   sub=good if rule!='tie_exclude' else [x for x in good if x['argmin_n']==1];key='offset' if rule=='tie_exclude' else rule
   sensitivity.append({'fold':fold,'rule':rule,'n':len(sub),'median_offset':float(np.median([x[key] for x in sub])) if sub else None,**{f'within{w}':sum(abs(x[key])<=w for x in sub) for w in [15,60,120,360]}})
  sensitivity.append({'fold':fold,'rule':'whole_near_zone_inside60','n':len(good),'within60':sum(x['near_low']>=-60 and x['near_high']<=60 for x in good)})
  sensitivity.append({'fold':fold,'rule':'near_zone_overlaps60','n':len(good),'within60':sum(x['near_low']<=60 and x['near_high']>=-60 for x in good)})
  RESULT.setdefault('zone_width',[]).append({'fold':fold,'q25':float(np.quantile([x['width'] for x in good],.25)),'median':float(np.median([x['width'] for x in good])),'q75':float(np.quantile([x['width'] for x in good],.75)),'exact_ties':sum(x['argmin_n']>1 for x in good)})
  cuts=np.quantile([x['hf_sd'] for x in b],[1/3,2/3])
  for kind in ['hf_sd','n']:
   for level in range(3):
    sub=[x for x in b if (int(np.searchsorted(cuts,x['hf_sd'],side='right')) if kind=='hf_sd' else (0 if x['n']==3 else 1 if x['n']<=5 else 2))==level]
    strata.append({'fold':fold,'variable':kind,'level':level,'n':len(sub),'identifiable':sum(x['ok'] for x in sub),'cuts':cuts.tolist() if kind=='hf_sd' else [3,5]})
  for rep in range(20):
   pp=[x for x in perm if x['fold']==fold and x['repetition']==rep];gg=[x for x in pp if x['ok']]
   null.append({'fold':fold,'rep':rep,'n':len(pp),'identifiable':len(gg),'within60':sum(abs(x['offset'])<=60 for x in gg)})
 RESULT['shifts']=summaries;RESULT['ties']=sensitivity;RESULT['strata']=strata;RESULT['permutations']=null
 RESULT['chart_rounding']=query(c,'SELECT p.fold,count(*),count(*) FILTER(WHERE minute(h.t)=0),count(*) FILTER(WHERE second(h.t)=0) FROM hf h JOIN primary_people p USING(subject_id) GROUP BY p.fold ORDER BY p.fold').fetchall()
 RESULT['boundary']=query(c,'SELECT fold,count(*),count(*) FILTER(WHERE ecg_time<intime OR ecg_time>outtime) FROM eligible GROUP BY fold ORDER BY fold').fetchall()
 c.close();assert sha(db)==MANIFEST[str(db)]['sha256'];save('offset_results.json',RESULT)

def matching():
 old=bind(str(data_path("s5.zaehlung_aw107_reproduktion_final_counts_duckdb"))); inv=read(bind(str(data_path("s5.s3_seriell_aw118_reproduktion_final_inventar_json"))))
 assert sha(old)==inv['projection_sha256'];c=connection();query(c,f"ATTACH '{old.as_posix()}' AS old (READ_ONLY)")
 assert query(c,"SELECT count(*) FROM old.whitelist WHERE fold NOT IN ('train','validation')").fetchone()[0]==0
 query(c,'CREATE TABLE cov(subject_id BIGINT,study_id BIGINT,hf DOUBLE,quality DOUBLE)')
 for fold in ['train','validation']:
  p=Path(f'{str(data_path("s5.mimic_validation_runs_a155_2026_10_04_d6e70e_analysis_feature_cache_cache_89d6aaf0a6b5957e"))}/split={fold}/features.parquet');pf=pq.ParquetFile(p)
  with p.open('rb') as f:f.seek(-8,2);n=int.from_bytes(f.read(4),'little');f.seek(-8-n,2);fh=hashlib.sha256(f.read(n+8)).hexdigest()
  expected=read(bind(str(data_path("s5.s3_seriell_aw118_reproduktion_final_quellen_hf_json"))))[fold];assert p.stat().st_size==expected['bytes'] and p.stat().st_mtime_ns==expected['mtime_ns'] and fh==expected['footer_sha256']
  MANIFEST[str(p)]={'bytes':p.stat().st_size,'footer_sha256':fh,'columns':['subject_id','study_id','context_heart_rate_bpm','signal_quality_score'],'partition':fold};save('input_manifest.json',MANIFEST)
  for batch in pf.iter_batches(columns=MANIFEST[str(p)]['columns'],batch_size=8192,use_threads=False):
   c.register('batch',batch);query(c,f"INSERT INTO cov SELECT b.subject_id,b.study_id,b.context_heart_rate_bpm,b.signal_quality_score FROM batch b JOIN old.whitelist w USING(subject_id,study_id) WHERE w.fold='{fold}'");guard()
 query(c,'CREATE TABLE states AS SELECT s.*,v.hf,v.quality FROM old.states s LEFT JOIN cov v USING(subject_id,study_id)')
 variants=[('base',.3,10,10,True,1),('K02',.2,10,10,True,1),('K05',.5,10,10,True,1),('K10',1.,10,10,True,1),('HR05',.3,5,10,True,1),('HR15',.3,15,10,True,1),('age05',.3,10,5,True,1),('age15',.3,10,15,True,1),('gap_relaxed',.3,10,10,False,1),('combined',1.,15,15,False,1),('two_controls',.3,10,10,True,2)]
 records=[];historical=read(bind(str(data_path("s5.s3_seriell_aw118_reproduktion_final_grenzpruefung_json"))))
 for w in [2,6]:
  query(c,f'''CREATE TABLE p{w} AS SELECT p.*,b.k k1,d.k k2,b.k_item,b.age,b.sex,b.k_sign sign1,d.k_sign sign2,b.hf hf1,d.hf hf2,b.quality quality1,d.quality quality2,b.machine_cart_id cart1,d.machine_cart_id cart2,
  CASE WHEN abs(d.k-b.k)>=1-1e-12 THEN 'Aenderung' WHEN abs(d.k-b.k)<=.2+1e-12 THEN 'Kontrolle' ELSE 'Zwischen' END arm
  FROM old.pairs p JOIN states b ON p.subject_id=b.subject_id AND p.s1=b.study_id JOIN states d ON p.subject_id=d.subject_id AND p.s2=d.study_id
  WHERE b.k_dt<={w} AND d.k_dt<={w} AND b.k BETWEEN 2.5 AND 6 AND d.k BETWEEN 2.5 AND 6 AND b.k_item=d.k_item AND b.k_specimen IS NOT NULL AND d.k_specimen IS NOT NULL AND b.k_specimen<>d.k_specimen AND b.k_time<d.k_time''')
  query(c,f'''CREATE TABLE support{w} AS SELECT * FROM p{w} WHERE isfinite(hf1) AND isfinite(hf2) AND hf1 BETWEEN 30 AND 180 AND hf2 BETWEEN 30 AND 180 AND quality1>=.6 AND quality2>=.6 AND (cart1 IS NULL OR cart2 IS NULL OR cart1=cart2)''')
  for fold in ['train','validation']:
   ee=[x[0] for x in query(c,f"SELECT DISTINCT subject_id FROM support{w} WHERE fold='{fold}' AND arm='Aenderung' ORDER BY 1").fetchall()];dd=[x[0] for x in query(c,f"SELECT DISTINCT subject_id FROM support{w} WHERE fold='{fold}' AND arm='Kontrolle' ORDER BY 1").fetchall()];ei={x:i for i,x in enumerate(ee)};di={x:i for i,x in enumerate(dd)}
   for name,k,hr,age,gap,capacity in variants:
    conditions=f"e.fold='{fold}' AND d.fold=e.fold AND e.subject_id<>d.subject_id AND e.arm='Aenderung' AND d.arm='Kontrolle' AND e.k_item=d.k_item AND coalesce(e.sex,'unbekannt')=coalesce(d.sex,'unbekannt') AND e.sign1=d.sign1 AND e.sign2=d.sign2 AND 5*(epoch_us(e.t2)-epoch_us(e.t1))>=4*(epoch_us(d.t2)-epoch_us(d.t1)) AND 4*(epoch_us(e.t2)-epoch_us(e.t1))<=5*(epoch_us(d.t2)-epoch_us(d.t1)) AND abs(cast(e.k1 AS DECIMAL(18,9))-cast(d.k1 AS DECIMAL(18,9)))<={k} AND abs(cast(e.hf1 AS DECIMAL(18,9))-cast(d.hf1 AS DECIMAL(18,9)))<={hr} AND abs(cast(e.age AS DECIMAL(18,9))-cast(d.age AS DECIMAL(18,9)))<={age}"
    if gap:conditions+=' AND (e.gap<6)=(d.gap<6) AND (e.gap<24)=(d.gap<24)'
    edges=query(c,f'SELECT DISTINCT e.subject_id,d.subject_id FROM support{w} e,support{w} d WHERE {conditions} ORDER BY 1,2').fetchall()
    combinations=query(c,f'SELECT count(*) FROM support{w} e,support{w} d WHERE {conditions}').fetchone()[0]
    graph=csr_matrix((np.ones(len(edges),dtype=np.int64),([ei[a] for a,b in edges],[di[b] for a,b in edges])),shape=(len(ee),len(dd)))
    card=int(sum(maximum_bipartite_matching(graph,perm_type='column')>=0))
    sink=len(ee)+len(dd)+1;arcs=[(0,i+1,capacity) for i in range(len(ee))]+[(ei[a]+1,len(ee)+di[b]+1,1) for a,b in edges]+[(len(ee)+i+1,sink,1) for i in range(len(dd))]
    flow=maximum_flow(csr_matrix(([v for a,b,v in arcs],([a for a,b,v in arcs],[b for a,b,v in arcs])),shape=(sink+1,sink+1)),0,sink)
    matched=int(flow.flow_value)
    if capacity==1:assert matched==card
    if name=='base':
     ref=next(x for x in historical['matching'] if x['window_h']==w and x['fold']==fold);assert (len(edges),matched)==(ref['edges'],ref['upper_bound_sets'])
    records.append({'window_h':w,'fold':fold,'variant':name,'K_caliper':k,'HR_caliper':hr,'age_caliper':age,'exact_gap_class':gap,'capacity':capacity,'change_people':len(ee),'control_people':len(dd),'episode_combinations':combinations,'person_edges':len(edges),'max_links':matched,'one_control_sets':card,'linked_change_people':int(sum(flow.flow[0,i+1]>0 for i in range(len(ee))))})
    print('Matching',w,fold,name,len(edges),matched,flush=True);guard()
 c.close();assert sha(old)==MANIFEST[str(old)]['sha256'];save('matching_results.json',records)

def af():
 pp=Path(str(data_path("s5.mimic_validation_runs_a155_2026_10_04_d6e70e_publication_preparation_research_documented_af_landmark_20261012_v2")))
 src=pp/'executed_source/a155_research_documented_af.py';cache=pp/'first-eligible-landmarks.parquet';seal=read(bind(pp/'completed-files.sha256.json'))
 assert sha(src)==seal['executed_source/a155_research_documented_af.py'] and sha(cache)==seal['first-eligible-landmarks.parquet']
 code=bind(pp/'executed_source/a155_research_af_landmark_revision002.py').read_text(encoding='utf-8-sig');assert "if not set(d.split.unique())<={'train','validation'}" in code
 ns={'np':np,'pd':pd,'re':re};pure(src,{'risk_set','report_sinus'},ns)
 pf=pq.ParquetFile(cache);schema=read(str(data_path("s5.aw178_af_schema_json")));assert sha(cache)==schema['sha256'];bind(cache)
 cols=['subject_id','split','age','sex','sex_male','ecg_time','dischtime','machine_report','machine_metadata_usable_for_primary','machine_heart_rate_bpm','first_af_event_delta_hours','next_af_event_delta_hours','next_af_event_source','af_dx_prior_admission','af_dx_index_admission','current_report_af_state','followup_hours','date_of_death','p_loop_width_major','rhythm_context']
 save('af_schema.json',{'rows':pf.metadata.num_rows,'columns':cols,'schema':str(pf.schema_arrow)})
 d=pf.read(columns=cols,use_threads=False).to_pandas();assert set(d.split.unique())<={'train','validation'} and not d.subject_id.duplicated().any()
 historical=read(bind(str(data_path("s5.aw178_r5_census_json"))));records=[]
 for fold in ['train','validation']:
  part=d.loc[d.split.eq(fold)]
  for horizon in [30,90,365]:
   risk,steps=ns['risk_set'](part,horizon);num=risk[['p_loop_width_major','age','sex_male','machine_heart_rate_bpm']].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan);cc=risk.loc[num.notna().all(axis=1)]
   for stage,frame in [('at_risk',risk),('complete_cases',cc)]:
    landmark=(pd.to_datetime(frame.dischtime)-pd.to_datetime(frame.ecg_time)).dt.total_seconds()/3600
    contact=pd.to_numeric(frame.followup_hours,errors='coerce')-landmark;death=(pd.to_datetime(frame.date_of_death)-pd.to_datetime(frame.dischtime)).dt.total_seconds()/3600
    ends=pd.DataFrame({'contact':contact,'death':death.fillna(np.inf),'horizon':horizon*24},index=frame.index);mi=ends.min(axis=1);tie=ends.eq(mi,axis=0).sum(axis=1)>1;no=frame.event.eq(0);dead=death.between(0,horizon*24,inclusive='right')
    r={'split':fold,'horizon_days':horizon,'stage':stage,'n':len(frame),'events':int(frame.event.sum()),'non_AF_contact_end':int((no&~tie&contact.eq(mi)).sum()),'non_AF_death_end':int((no&~tie&death.eq(mi)).sum()),'non_AF_horizon_end':int((no&~tie&ends.horizon.eq(mi)).sum()),'non_AF_tied_end':int((no&tie).sum()),'death_recorded_before_horizon':int(dead.sum()),'death_after_contact_before_horizon':int((dead&death.gt(contact)).sum()),'death_before_horizon_after_AF':int((dead&frame.event.eq(1)).sum()),'contact_fraction_no_AF':float((no&~tie&contact.eq(mi)).sum()/no.sum()),'selection_steps':steps}
    assert r['events']+sum(r[k] for k in ['non_AF_contact_end','non_AF_death_end','non_AF_horizon_end','non_AF_tied_end'])==r['n']
    if stage=='complete_cases':
     ref=next(x for x in historical if x['split']==fold and x['horizon_days']==horizon and x['stage']==stage)
     for k in ['n','events','non_AF_contact_end','non_AF_death_end','non_AF_horizon_end','non_AF_tied_end','death_recorded_before_horizon']:assert r[k]==ref[k],k
    records.append(r)
 save('af_results.json',records);print('AF census reproduced',flush=True)

def anchors():
 old=bind(str(data_path("s5.zaehlung_aw107_reproduktion_final_counts_duckdb")));c=connection();query(c,f"ATTACH '{old.as_posix()}' AS old (READ_ONLY)")
 assert query(c,"SELECT count(*) FROM old.whitelist WHERE fold NOT IN ('train','validation')").fetchone()[0]==0
 p=Path(str(data_path("s5.mimic_validation_standardized_admissions_parquet")));pf=pq.ParquetFile(p)
 cols=['subject_id','hadm_id','admittime','dischtime'];assert set(cols)<=set(pf.schema_arrow.names)
 MANIFEST[str(p)]={'bytes':p.stat().st_size,'columns':cols,'schema':str(pf.schema_arrow),'read_rule':'join assigned admissions to development-only old.ecg'};save('input_manifest.json',MANIFEST)
 sql=f'''SELECT e.fold,count(*) linked_ecgs,count(*) FILTER(WHERE a.admittime IS NULL OR a.dischtime IS NULL) missing_boundaries,count(*) FILTER(WHERE e.ecg_time<a.admittime OR e.ecg_time>a.dischtime) nominal_outside,count(DISTINCT e.subject_id) people FROM old.ecg e JOIN read_parquet('{p.as_posix()}') a ON e.subject_id=a.subject_id AND e.hadm_id=a.hadm_id WHERE e.fold IN ('train','validation') GROUP BY e.fold ORDER BY e.fold'''
 data=query(c,sql).fetchall();save('admission_boundary.json',{'columns':[x[0] for x in c.description],'rows':data,'interpretation':'Conditional on stored admission link, not all ECGs or independent clock truth; nominal linkage can select away discordance.'});c.close()

def main():
 start=time.monotonic();save('execution_start.json',{'pid':os.getpid(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_sha256':PROTOCOL,'script_sha256':sha(__file__),'estimated_peak_gib':1.2,'free_ram_gib':psutil.virtual_memory().available/2**30})
 phase=sys.argv[1]
 {'offset':offset,'matching':matching,'af':af,'anchors':anchors}[phase]()
 (OUT/('queries_'+phase+'.sql')).write_text(';\n'.join(SQL),encoding='utf-8')
 save('execution_'+phase+'.json',{'pid':os.getpid(),'elapsed_seconds':time.monotonic()-start,'peak_working_set_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None),'test_access':False,'protocol_sha256':PROTOCOL,'script_sha256':sha(__file__),'input_manifest_sha256':sha(OUT/'input_manifest.json')})
 print('Phase finished',phase,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  (OUT/'compute_error_private.txt').write_text(traceback.format_exc(),encoding='utf-8');print('Error: check the private error file',flush=True);sys.exit(1)
