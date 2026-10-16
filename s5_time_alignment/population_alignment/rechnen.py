"""AW110: begrenzter Zeit-/HF-Audit, ausschließlich freigegebene Entwicklung."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os,sys,json,time,hashlib,threading,csv
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
os.environ.update({x:'1' for x in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
import psutil
PEAK=0
def watch():
 global PEAK
 while True:
  PEAK=max(PEAK,psutil.Process().memory_info().rss)
  if PEAK>1.5*1024**3: os._exit(75)
  time.sleep(.1)
threading.Thread(target=watch,daemon=True).start()
import duckdb,pyarrow.parquet as pq,numpy as np
ROOT=Path(__file__).resolve().parents[2]; B=Path(str(data_path("s5.mimic_validation"))); O=Path(str(data_path("s5.uhrzeit_aw110")));O.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1<<20),b''): h.update(v)
 return h.hexdigest()
SPEC=ROOT/'nachrechnung/SPEZIFIKATION_AW110_UHRZEIT.md'
assert sha(SPEC)=='a5412cbdd2df759b462a214c59fc6028d70c2811d10653ede75c19fffde34a76'
lock=O/'spec_lock.json'
if not lock.exists():lock.write_text(json.dumps({'sha256':sha(SPEC),'locked_utc':datetime.now(timezone.utc).isoformat()}),encoding='utf-8')
c=duckdb.connect(str(O/'audit.duckdb'));c.execute("SET memory_limit='700MB'; SET threads=2");c.execute(f"SET temp_directory='{O.as_posix()}/spill'")
SOURCES={};SQL=[];R={}
def footer(p):
 p=Path(p);f=pq.ParquetFile(p)
 with p.open('rb') as s:
  s.seek(-8,2);n=int.from_bytes(s.read(4),'little');s.seek(-8-n,2);h=hashlib.sha256(s.read(n+8)).hexdigest()
 SOURCES[str(p)]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'footer_sha256':h,'columns':f.schema_arrow.names}
 return f
def source(n):
 p=B/f'standardized/{n}.parquet';footer(p);return f"read_parquet('{p.as_posix()}')"
def sql(q):SQL.append(q);return c.execute(q)
def agg(k,q):
 a=sql(q);R[k]={'columns':[x[0] for x in a.description],'rows':a.fetchall()}
def save():
 (O/'sources.json').write_text(json.dumps(SOURCES,indent=2),encoding='utf-8')
 (O/'queries.sql').write_text(';\n'.join(SQL),encoding='utf-8')
 (O/'prepare.json').write_text(json.dumps(R,indent=2,default=str),encoding='utf-8')
def prepare():
 assert not c.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='whitelist'").fetchone()[0], 'Vorhandene Basis nicht überschreiben'
 sql('CREATE TABLE whitelist(subject_id BIGINT,study_id BIGINT,fold VARCHAR)')
 for fold in ['train','validation']:
  p=B/f'runs/A155-2026-10-04-d6e70e/analysis_feature_cache/cache-89d6aaf0a6b5957e/split={fold}/features.parquet'
  for batch in footer(p).iter_batches(columns=['subject_id','study_id'],batch_size=8192,use_threads=False):
   c.register('batch',batch);sql(f"INSERT INTO whitelist SELECT subject_id,study_id,'{fold}' FROM batch")
 assert sql('SELECT count(*) FROM (SELECT subject_id FROM whitelist GROUP BY subject_id HAVING count(DISTINCT fold)>1)').fetchone()[0]==0
 assert sql('SELECT count(*) FROM (SELECT subject_id,study_id FROM whitelist GROUP BY ALL HAVING count(*)>1)').fetchone()[0]==0
 p=B/'master/ecg_master_index.parquet';footer(p)
 sql(f"CREATE TABLE ecg AS SELECT m.subject_id,m.study_id,m.ecg_time,m.anchor_year_group,m.machine_cart_id,m.ed_stay_id,m.ed_intime,m.ed_outtime,w.fold FROM read_parquet('{p.as_posix()}') m JOIN whitelist w USING(subject_id,study_id) WHERE m.split=w.fold")
 icu=source('icustays');rr=source('machine_measurements')
 sql(f'''CREATE TABLE icu_matches AS SELECT e.*,i.icustay_id,i.first_careunit,i.intime,i.outtime,count(*) OVER(PARTITION BY e.subject_id,e.study_id) candidates FROM ecg e JOIN {icu} i ON e.subject_id=i.subject_id AND e.ecg_time BETWEEN i.intime AND i.outtime''')
 sql('CREATE TABLE primary_ecg AS SELECT * FROM icu_matches WHERE candidates=1')
 sql('CREATE TABLE primary_people AS SELECT DISTINCT subject_id,fold FROM primary_ecg')
 sql(f'''CREATE TABLE rr_ecg AS SELECT e.*,60000/m.rr_interval_ms hf FROM ecg e JOIN {rr} m USING(subject_id,study_id) WHERE m.rr_interval_ms>0 AND isfinite(m.rr_interval_ms) AND 60000/m.rr_interval_ms BETWEEN 30 AND 220 AND abs(epoch(m.machine_ecg_time-e.ecg_time))<=60''')
 sql('CREATE TABLE eligible AS SELECT i.*,r.hf FROM primary_ecg i JOIN rr_ecg r USING(subject_id,study_id)')
 sql('CREATE INDEX eligible_people ON eligible(subject_id)')
 sql('CREATE INDEX rr_people ON rr_ecg(subject_id)')
 di=source('d_items')
 ids=sql(f"SELECT itemid FROM {di} WHERE lower(label)='heart rate' AND lower(linksto)='chartevents'").fetchall()
 assert ids,'HF-ID fehlt'
 R['hf_items']=ids
 agg('anchor_dictionary',f"SELECT itemid,label,linksto FROM {di} WHERE regexp_matches(lower(label),'rhythm|cardioversion|pacer|pacemaker|pacing') ORDER BY itemid")
 agg('basis','SELECT fold,count(*) ecgs,count(DISTINCT subject_id) personen FROM ecg GROUP BY fold ORDER BY fold')
 agg('icu','SELECT fold,count(*) ecgs,count(DISTINCT subject_id) personen FROM primary_ecg GROUP BY fold ORDER BY fold')
 agg('icu_ambig','SELECT fold,count(DISTINCT study_id) ecgs FROM icu_matches WHERE candidates>1 GROUP BY fold')
 agg('rr','SELECT fold,count(*) ecgs,count(DISTINCT subject_id) personen FROM eligible GROUP BY fold ORDER BY fold')
 sql('CREATE TABLE hf_raw(subject_id BIGINT,icustay_id BIGINT,t TIMESTAMP,hf DOUBLE)')
 ce=B/'standardized/clinical_events/versions/schema-1.2.0-922758943bbed5d1/chartevents'
 files=sorted(ce.rglob('*.parquet'))
 for ix,p in enumerate(files):
  footer(p)
  sql(f'''INSERT INTO hf_raw SELECT h.subject_id,h.icustay_id,h.charttime,h.valuenum FROM read_parquet('{p.as_posix()}') h JOIN primary_people w USING(subject_id) WHERE h.itemid IN ({','.join(str(v[0]) for v in ids)}) AND h.charttime IS NOT NULL AND isfinite(h.valuenum) AND h.valuenum BETWEEN 30 AND 220''')
  if ix%50==0:print(json.dumps({'chartevents_files_done':ix,'total':len(files)}),flush=True)
 sql('CREATE TABLE hf AS SELECT subject_id,icustay_id,t,median(hf) hf FROM hf_raw GROUP BY ALL')
 sql('CREATE INDEX hf_people ON hf(subject_id)')
 vs=source('vitalsign');source('triage')
 sql(f'''CREATE TABLE edhf AS SELECT h.subject_id,h.ed_stay_id,h.charttime t,median(h.heart_rate) hf FROM {vs} h JOIN (SELECT DISTINCT subject_id FROM rr_ecg) w USING(subject_id) WHERE h.charttime IS NOT NULL AND isfinite(h.heart_rate) AND h.heart_rate BETWEEN 30 AND 220 GROUP BY ALL''')
 sql('CREATE INDEX ed_people ON edhf(subject_id)')
 agg('hf','SELECT w.fold,count(*) events,count(DISTINCT h.subject_id) personen FROM hf h JOIN primary_people w USING(subject_id) GROUP BY w.fold ORDER BY w.fold')
 save()

D=np.arange(-720,721,5); SHIFT=D*60
def curves(ecgs,charts):
 losses=[];metas=[];zero=[];distances=[]
 for ecg in ecgs:
  # time represented as seconds, reference dictionary keyed by ICU/ED stay
  study,stay,t,hf,year,unit,cart,start,end=ecg
  if stay not in charts:continue
  times,values=charts[stay];targets=t+SHIFT
  inside=(times>=start)&(times<=end);times=times[inside];values=values[inside]
  if not len(times):continue
  k=np.searchsorted(times,targets);left=np.clip(k-1,0,len(times)-1);right=np.clip(k,0,len(times)-1)
  pick=np.where(abs(times[left]-targets)<=abs(times[right]-targets),left,right)
  dt=abs(times[pick]-targets);err=abs(values[pick]-hf)
  if dt[144]<=1800:zero.append(float(err[144]))
  if np.all((dt<=1800)&(targets>=start)&(targets<=end)):losses.append(err);metas.append((year,unit,cart));distances.extend(dt.tolist())
 return losses,metas,zero,distances
def charts(rows):
 a={}
 for stay,t,h in rows:a.setdefault(stay,[]).append((t,h))
 return {k:(np.array(v)[:,0],np.array(v)[:,1]) for k,v in a.items()}
def estimate(L):
 loss=np.median(L,axis=0);lo=np.min(loss);near=D[loss<=lo+1];opts=D[np.isclose(loss,lo,rtol=0,atol=1e-10)]
 offset=sorted(opts,key=lambda x:(abs(x),x))[0];width=near[-1]-near[0]
 ok=len(L)>=3 and width<=120 and lo<=5 and near[0]>-720 and near[-1]<720 and np.all(np.diff(near)==5)
 return loss,int(offset),int(width),bool(ok),float(lo)
def calc(mode='icu'):
 c.execute('CREATE INDEX IF NOT EXISTS eligible_people ON eligible(subject_id)')
 c.execute('CREATE INDEX IF NOT EXISTS rr_people ON rr_ecg(subject_id)')
 results=[];pop={'train':[],'validation':[]};close={'train':[],'validation':[]};zero={'train':[],'validation':[]};dist=np.zeros(1801,dtype=np.int64)
 if mode=='icu':
  people=c.execute('SELECT subject_id,fold FROM primary_people ORDER BY subject_id').fetchall()
 else:people=c.execute('SELECT DISTINCT subject_id,fold FROM rr_ecg WHERE ed_stay_id IS NOT NULL AND ecg_time BETWEEN ed_intime AND ed_outtime ORDER BY subject_id').fetchall()
 denominators={f:sum(f==g for _,g in people) for f in ['train','validation']}
 if mode=='ed':
  c.execute('''CREATE OR REPLACE TEMP TABLE ed_potential AS SELECT e.* FROM rr_ecg e JOIN (SELECT subject_id,ed_stay_id,min(t) mint,max(t) maxt FROM edhf GROUP BY ALL) h USING(subject_id,ed_stay_id) WHERE e.ecg_time BETWEEN e.ed_intime+INTERVAL '12 hour' AND e.ed_outtime-INTERVAL '12 hour' AND h.mint<=e.ecg_time-INTERVAL '11.5 hour' AND h.maxt>=e.ecg_time+INTERVAL '11.5 hour' ''')
  people=c.execute('SELECT DISTINCT subject_id,fold FROM ed_potential ORDER BY subject_id').fetchall()
 for ix,(person,fold) in enumerate(people):
  if mode=='icu':
   e=c.execute('SELECT study_id,icustay_id,epoch(ecg_time),hf,anchor_year_group,first_careunit,machine_cart_id,epoch(intime),epoch(outtime) FROM eligible WHERE subject_id=? ORDER BY study_id',[person]).fetchall()
   h=c.execute('SELECT icustay_id,epoch(t),hf FROM hf WHERE subject_id=? ORDER BY icustay_id,t',[person]).fetchall()
  else:
   e=c.execute("SELECT study_id,ed_stay_id,epoch(ecg_time),hf,anchor_year_group,'ED',machine_cart_id,epoch(ed_intime),epoch(ed_outtime) FROM ed_potential WHERE subject_id=? ORDER BY study_id",[person]).fetchall()
   h=c.execute('SELECT ed_stay_id,epoch(t),hf FROM edhf WHERE subject_id=? ORDER BY ed_stay_id,t',[person]).fetchall()
  L,meta,z,d=curves(e,charts(h))
  if d:dist+=np.bincount(np.rint(d).astype(int),minlength=1801)
  if z:zero[fold].append(float(np.median(z)))
  if L:
   loss,off,width,ok,minimum=estimate(L);pop[fold].append(loss);close[fold].append(np.mean(np.array(L)<=5,axis=0))
   strata=[]
   for j in range(3):
    counts={str(v[j]) if v[j] is not None else 'unbekannt':0 for v in meta}
    for v in meta:counts[str(v[j]) if v[j] is not None else 'unbekannt']+=1
    strata.append(sorted(counts,key=lambda k:(-counts[k],k))[0])
   results.append({'fold':fold,'n':len(L),'offset':off,'width':width,'ok':ok,'minimum':minimum,'year':strata[0],'unit':strata[1],'cart':strata[2]})
  if ix%1000==0:print(json.dumps({'mode':mode,'people_done':ix,'total':len(people),'peak_rss':PEAK}),flush=True)
 out={'mode':mode,'folds':{},'strata':{},'reference_distance_seconds_quantiles':[int(np.searchsorted(np.cumsum(dist),q*dist.sum())) for q in [.5,.9,.95,1]] if dist.sum() else []}
 rng=np.random.default_rng(110)
 def summary(rows,denom,infer=True):
  good=[r for r in rows if r['ok']];offs=np.array([r['offset'] for r in good]);widths=np.array([r['width'] for r in good]);n=len(offs)
  s={'primary_people':denom,'complete_people':len(rows),'complete_ecgs':sum(r['n'] for r in rows),'at_least_3_people':sum(r['n']>=3 for r in rows),'identifiable_people':n,'unrestricted_offset_median':float(np.median([r['offset'] for r in rows])) if rows else None}
  if n:
   s.update({'offset_median':float(np.median(offs)),'width_median':float(np.median(widths)),'offset_quantiles':np.quantile(offs,[.025,.1,.5,.9,.975]).tolist(),'within':{str(w):{'count':int(sum(abs(offs)<=w)),'conditional_fraction':float(np.mean(abs(offs)<=w)),'primary_fraction':float(sum(abs(offs)<=w)/denom)} for w in [15,60,360,720]}})
   s['criterion_conditional']=bool(abs(s['offset_median'])<=15 and s['within']['60']['conditional_fraction']>=.9)
   s['criterion_primary']=bool(abs(s['offset_median'])<=15 and s['within']['60']['primary_fraction']>=.9)
   if not infer:return s
   boot=[]
   # Bootstrap includes unavailable primary persons in primary-fraction denominator.
   alloff=np.r_[offs,np.full(denom-n,np.nan)]
   for b in range(1000):
    a=alloff[rng.integers(0,denom,denom)];v=a[np.isfinite(a)]
    boot.append([np.median(v) if len(v) else np.nan,np.mean(abs(v)<=60) if len(v) else np.nan,np.sum(abs(v)<=60)/denom])
   s['bootstrap_95ci']=dict(zip(['median','conditional_within60','primary_within60'],np.nanquantile(boot,[.025,.975],axis=0).T.tolist()))
   s['width_bootstrap_95ci']=np.quantile([np.median(widths[rng.integers(0,n,n)]) for _ in range(1000)],[.025,.975]).tolist()
   s['criterion_conditional']=bool(abs(s['offset_median'])<=15 and s['within']['60']['conditional_fraction']>=.9)
   s['criterion_primary']=bool(abs(s['offset_median'])<=15 and s['within']['60']['primary_fraction']>=.9)
  return s
 for fold in ['train','validation']:
  sub=[r for r in results if r['fold']==fold];denom=denominators[fold]
  s=summary(sub,denom);s['zero_available_people']=len(zero[fold]);s['zero_person_median_abs_error']=float(np.median(zero[fold])) if zero[fold] else None
  if pop[fold]:
   curve=np.median(pop[fold],axis=0);s['population_curve']=curve.tolist();s['population_optimum_minutes']=int(D[np.argmin(curve)]);s['mean_person_fraction_within5_curve']=np.mean(close[fold],axis=0).tolist()
  out['folds'][fold]=s
  for kind in ['year','unit','cart']:
   groups={}
   for r in sub:groups.setdefault(r[kind],[]).append(r)
   small=[r for v in groups.values() if len(v)<30 for r in v]
   groups={k:v for k,v in groups.items() if len(v)>=30}
   if small:groups['unter_30']=small
   out['strata'][fold+'_'+kind]={k:summary(v,len(v),False) for k,v in groups.items()}
 (O/f'{mode}_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 # No identifiers, only person-level metrics retained privately for reproducibility.
 (O/f'{mode}_anonymous_metrics.json').write_text(json.dumps(results),encoding='utf-8')
 print(json.dumps({f:{k:v for k,v in s.items() if not k.endswith('curve')} for f,s in out['folds'].items()}),flush=True)
if __name__=='__main__':
 stage=sys.argv[1]
 if stage=='prepare':prepare()
 elif stage in ['icu','ed']:calc(stage)
 else:raise ValueError(stage)
 (O/f'resource_{stage}.json').write_text(json.dumps({'peak_rss_bytes':PEAK,'code_sha256':sha(__file__),'spec_sha256':sha(SPEC),'completed_utc':datetime.now(timezone.utc).isoformat()}),encoding='utf-8')
 c.close()
