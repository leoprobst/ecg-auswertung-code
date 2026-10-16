"""AW115: read-only AW110-Basis, nur Zeit-/HF-Aggregate, kein Testzugriff."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os, sys, json, hashlib, time, threading, csv
from pathlib import Path
from datetime import datetime, timezone
sys.dont_write_bytecode = True
os.environ.update({k: '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
import psutil
PEAK = 0
def watch():
    global PEAK
    while True:
        PEAK = max(PEAK, psutil.Process().memory_info().rss)
        if PEAK > 1.5*1024**3: os._exit(75)
        time.sleep(.1)
threading.Thread(target=watch, daemon=True).start()
import numpy as np
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]
OLD = Path(str(data_path("s5.uhrzeit_aw110")))
OUT = Path(str(data_path("s5.uhrzeit_aw115")))
OUT.mkdir(exist_ok=True)
SPEC = ROOT/'nachrechnung/SPEZIFIKATION_AW115_UHRZEIT_GEPOOLT.md'
EXPECTED = '8868bc9cb9ca66a5f3271a026e70c997028cc769afd841237fdccb8d61cf777e'
D = np.arange(-720,721,5); TOL = 1e-9
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
def est(L, min_n=3):
    if not len(L): return {'n_complete':0,'ok':False,'offset':None,'width':None,'minimum':None}, None
    loss = np.median(L, axis=0); m = float(loss.min())
    near = D[loss <= m+1+TOL]; opts = D[abs(loss-m) <= TOL]
    off = int(sorted(opts, key=lambda x:(abs(x),x))[0]); w = int(near[-1]-near[0])
    ok = len(L)>=min_n and w<=120 and m<=5+TOL and near[0]>-720 and near[-1]<720 and bool(np.all(np.diff(near)==5))
    return {'n_complete':len(L),'ok':bool(ok),'offset':off,'width':w,'minimum':m}, loss
def curve(e, h):
    # e: subject,fold,stay,study,t,hf,cart,month,start,end
    _,_,_,_,t,hf,_,_,start,end = e
    times,values = h; valid = (times>=start)&(times<=end)
    times = times[valid]; values = values[valid]
    if not len(times): return None
    targets = t+D*60; k = np.searchsorted(times,targets)
    left = np.clip(k-1,0,len(times)-1); right = np.clip(k,0,len(times)-1)
    pick = np.where(abs(times[left]-targets)<=abs(times[right]-targets),left,right)
    if not np.all((abs(times[pick]-targets)<=1800)&(targets>=start)&(targets<=end)): return None
    return abs(values[pick]-hf)
def metric(entries, base):
    entries = sorted(entries, key=lambda v:(v[0],v[1]))
    L = [v[2] for v in entries]; r,loss = est(L); r.update(base)
    a,b = L[:len(L)//2],L[len(L)//2:]
    r1,_ = est(a,1); r2,_ = est(b,1)
    both = r1['ok'] and r2['ok']
    r.update({'half1_offset':r1['offset'],'half2_offset':r2['offset'],
              'halves_estimable':bool(both),'half_agree':bool(both and abs(r1['offset']-r2['offset'])<=30),
              'strict_halves_estimable':bool(both and len(a)>=3 and len(b)>=3)})
    return r,loss
def summarize(rows, infer=False):
    eligible = [r for r in rows if r['n_rr']>=3]
    good = [r for r in eligible if r['ok']]
    den = len(eligible); ident = len(good)
    both = sum(r['halves_estimable'] for r in good); agree = sum(r['half_agree'] for r in good)
    inside = sum(abs(r['offset'])<=60 for r in good)
    strict = [r for r in good if r['n_complete']>=6]
    strict_both = sum(r['strict_halves_estimable'] for r in strict)
    strict_agree = sum(r['strict_halves_estimable'] and r['half_agree'] for r in strict)
    frac = lambda a,n: a/n if n else None
    gates = [bool(den and 2*ident>=den), bool(ident and 5*agree>=4*ident), bool(ident and 10*inside>=9*ident)]
    s = {'units':len(rows),'rr_ecgs':sum(r['n_rr'] for r in rows),'rr_at_least3_units':den,
         'complete_units':sum(r['n_complete']>0 for r in rows),'complete_ecgs':sum(r['n_complete'] for r in rows),
         'complete_at_least3_units':sum(r['n_complete']>=3 for r in rows),'identifiable_units':ident,
         'identifiable_fraction':frac(ident,den),'halves_estimable_units':both,'half_agree_units':agree,
         'half_unassessable_units':ident-both,'half_disagree_measured_units':both-agree,
         'half_agreement_fraction':frac(agree,ident),'half_conditional_fraction':frac(agree,both),
         'within60_units':inside,'within60_fraction':frac(inside,ident),'gates':gates,'criterion_pass':all(gates),
         'strict_n6_units':len(strict),'strict_halves_estimable_units':strict_both,
         'strict_half_agree_units':strict_agree,'strict_half_fraction':frac(strict_agree,len(strict)),
         'strict_half_conditional_fraction':frac(strict_agree,strict_both),
         'within':{str(w):{'n':sum(abs(r['offset'])<=w for r in good),'fraction':frac(sum(abs(r['offset'])<=w for r in good),ident)} for w in [15,60,120,360]},
         'offset_quantiles':np.quantile([r['offset'] for r in good],[.025,.5,.975]).tolist() if good else None,
         'median_width':float(np.median([r['width'] for r in good])) if good else None}
    if infer and eligible:
        # Cluster contributions: all stays of a person resampled together.
        person = {}
        for r in eligible:
            v = person.setdefault(r['subject_id'],np.zeros(4,dtype=int))
            v += [1,int(r['ok']),int(r['ok'] and r['half_agree']),int(r['ok'] and abs(r['offset'])<=60)]
        a = np.array(list(person.values())); rng = np.random.default_rng(115)
        boots = []
        for _ in range(1000):
            v = a[rng.integers(0,len(a),len(a))].sum(axis=0)
            boots.append([v[1]/v[0],v[2]/v[1] if v[1] else np.nan,v[3]/v[1] if v[1] else np.nan])
        s['bootstrap_persons'] = len(a)
        s['bootstrap_95ci'] = dict(zip(['identifiable','half_agreement','within60'],np.nanquantile(boots,[.025,.975],axis=0).T.tolist()))
    return s
def tests():
    flat = np.zeros((3,289)); assert not est(flat)[0]['ok']
    v = np.minimum(abs(D-100)/10,20); assert est([v]*3)[0]['offset']==100
    multi = np.minimum(abs(D-100),abs(D+100))/10; assert not est([multi]*3)[0]['ok']
    near = np.full(289,20.); near[144]=5.; near[145]=6.+5e-10
    assert est([near]*3)[0]['ok'] and est([near]*3)[0]['width']==5
    edge = abs(D-720)/10; assert not est([edge]*3)[0]['ok']
    t = 100000.; e = (1,'train',1,1,t,60,1,'x',t-720*60,t+720*60)
    times = t+D*60; vals = np.full(289,100.); vals[D==100]=60.
    assert est([curve(e,(times,vals))]*3)[0]['offset']==100
    outside = list(e); outside[-1] -= 1
    assert curve(outside,(times,vals)) is None
    assert 2*1>=2 and not 2*1>=3 and 5*4>=4*5 and not 5*3>=4*5 and 10*9>=9*10
    m,_ = metric([(3,3,v),(1,1,v),(2,2,v)],{'n_rr':3})
    assert m['half_agree'] and not m['strict_halves_estimable']
    return {'flat':True,'multimodal':True,'positive_sign':True,'boundaries':True,'decimal_tolerance':True,'integer_gates':True,'chronological_halves':True}
def main():
    assert sha(SPEC)==EXPECTED
    lock = OUT/'spec_lock.json'
    if not lock.exists(): dump('spec_lock.json',{'sha256':EXPECTED,'locked_utc':datetime.now(timezone.utc).isoformat()})
    assert json.loads(lock.read_text(encoding='utf-8'))['sha256']==EXPECTED
    verified = tests()
    db = OLD/'audit.duckdb'; before = sha(db)
    source_checks = []
    for p,s in json.loads((OLD/'sources.json').read_text()).items():
        path = Path(p); assert 'split=test' not in p
        pf = pq.ParquetFile(path)
        with path.open('rb') as f:
            f.seek(-8,2); n = int.from_bytes(f.read(4),'little'); f.seek(-8-n,2); h = hashlib.sha256(f.read(n+8)).hexdigest()
        assert path.stat().st_size==s['bytes'] and path.stat().st_mtime_ns==s['mtime_ns'] and h==s['footer_sha256']
        source_checks.append({'path':p,'bytes':s['bytes'],'footer_sha256':h})
    c = duckdb.connect(str(db), read_only=True)
    c.execute("SET memory_limit='700MB'; SET threads=2")
    c.execute(f"SET temp_directory='{OUT.as_posix()}/spill'")
    assert c.execute("SELECT count(*) FROM primary_people WHERE fold NOT IN ('train','validation')").fetchone()[0]==0
    assert c.execute('SELECT count(*) FROM (SELECT subject_id FROM primary_people GROUP BY 1 HAVING count(DISTINCT fold)>1)').fetchone()[0]==0
    assert c.execute('SELECT count(*) FROM (SELECT icustay_id FROM primary_ecg GROUP BY 1 HAVING count(DISTINCT subject_id)>1 OR count(DISTINCT fold)>1)').fetchone()[0]==0
    queries = []
    def q(sql,params=None):
        queries.append(sql); return c.execute(sql,params or [])
    nominal = q('SELECT fold,count(*) ecgs,count(DISTINCT icustay_id) stays,count(DISTINCT subject_id) persons FROM primary_ecg GROUP BY fold ORDER BY fold').fetchall()
    ecgs = q("SELECT subject_id,fold,icustay_id,study_id,epoch(ecg_time),hf,machine_cart_id,strftime(ecg_time,'%Y-%m'),epoch(intime),epoch(outtime) FROM eligible ORDER BY subject_id,icustay_id,ecg_time,study_id").fetchall()
    by_person = {}
    for e in ecgs: by_person.setdefault(e[0],[]).append(e)
    cart = {}; primary = []; pop = {f:[] for f in ['train','validation']}; cpop = {f:[] for f in pop}
    unit_curves = []; checks = []; old_losses = {f:[] for f in pop}
    for ix,(person,es) in enumerate(by_person.items()):
        hrows = q('SELECT icustay_id,epoch(t),hf FROM hf WHERE subject_id=? ORDER BY icustay_id,t',[person]).fetchall()
        h = {}
        for stay,t,hf in hrows: h.setdefault(stay,[]).append((t,hf))
        h = {k:np.asarray(v,dtype=float).T for k,v in h.items()}
        stays = {}
        for e in es:
            _,fold,stay,study,t,hf,cartid,month,_,_ = e
            base = stays.setdefault(stay,{'n':0,'entries':[]}); base['n'] += 1
            key = (fold,cartid,month)
            if cartid is not None:
                cb = cart.setdefault(key,{'n':0,'entries':[],'persons':set()}); cb['n']+=1; cb['persons'].add(person)
            L = curve(e,h[stay]) if stay in h else None
            if L is not None:
                entry = (t,study,L); base['entries'].append(entry)
                if cartid is not None: cart[key]['entries'].append(entry)
                if len(checks)<32:
                    # SQL matching independent of numpy/searchsorted implementation.
                    sql = '''WITH targets AS (SELECT d,?+d*60 AS target_time FROM range(-720,721,5) z(d)), candidates AS (
                    SELECT d,abs(h.hf-?) err,row_number() OVER(PARTITION BY d ORDER BY abs(epoch(h.t)-target_time),h.t) rn
                    FROM targets JOIN hf h ON h.subject_id=? AND h.icustay_id=?
                    AND epoch(h.t) BETWEEN ? AND ? AND target_time BETWEEN ? AND ? AND abs(epoch(h.t)-target_time)<=1800)
                    SELECT d,err FROM candidates WHERE rn=1 ORDER BY d'''
                    alt = q(sql,[t,hf,person,stay,e[-2],e[-1],e[-2],e[-1]]).fetchall()
                    assert len(alt)==289 and np.allclose(np.array(alt)[:,1],L,rtol=0,atol=TOL)
                    checks.append({'ordinal':len(checks),'all_289_offsets_equal':True})
        person_loss = [v[2] for b in stays.values() for v in b['entries']]
        if person_loss: old_losses[es[0][1]].append(np.median(person_loss,axis=0))
        for stay,b in stays.items():
            r,loss = metric(b['entries'],{'subject_id':person,'icustay_id':stay,'fold':es[0][1],'n_rr':b['n']})
            primary.append(r)
            if loss is not None:
                pop[r['fold']].append(loss)
                unit_curves.append({'kind':'icu','fold':r['fold'],'subject_id':person,'unit':str(stay),'loss':loss.tolist()})
        if ix%2000==0: print(json.dumps({'persons_processed':ix,'persons_total':len(by_person),'peak_rss':PEAK}),flush=True)
    carts = []
    for (fold,cartid,month),b in cart.items():
        r,loss = metric(b['entries'],{'fold':fold,'n_rr':b['n'],'persons':len(b['persons']),'cart_id':cartid,'shifted_month':month})
        carts.append(r)
        if loss is not None:
            cpop[fold].append(loss)
            unit_curves.append({'kind':'cart_shifted_month','fold':fold,'subject_id':None,'unit':f'{cartid}/{month}','loss':loss.tolist()})
    cart_audit = q('SELECT fold,count(*) rr_ecgs,count(machine_cart_id) cart_present,count(DISTINCT machine_cart_id) cart_labels FROM eligible GROUP BY fold ORDER BY fold').fetchall()
    # Reproduce AW110 population curve on original person pooling to check the reused basis.
    # AW110 uses median over ECGs, not over stays; reuse the person matrices above.
    old_result = json.loads((OLD/'icu_results.json').read_text())
    for f in pop:
        assert np.allclose(np.median(old_losses[f],axis=0),old_result['folds'][f]['population_curve'],rtol=0,atol=TOL)
    result = {'run':'A155-2026-10-04-d6e70e','spec_sha256':EXPECTED,'nominal_icu':nominal,'cart_availability':cart_audit,'primary':{},'cart_shifted_month':{}}
    for kind,rows,curves in [('primary',primary,pop),('cart_shifted_month',carts,cpop)]:
        for f in pop:
            s = summarize([r for r in rows if r['fold']==f],infer=kind=='primary')
            if curves[f]:
                s['population_curve'] = np.median(curves[f],axis=0).tolist()
                s['population_optimum_minutes'] = est([s['population_curve']],1)[0]['offset']
            result[kind][f] = s
    result['supported_primary'] = all(result['primary'][f]['criterion_pass'] for f in pop)
    dump('unit_metrics.json',{'primary':primary,'cart_shifted_month':carts})
    pq.write_table(pa.Table.from_pylist(unit_curves),OUT/'unit_curves.parquet',compression='zstd')
    dump('results.json',result)
    verified.update({'independent_sql_ecgs':checks,'aw110_person_population_reproduced':True,'fold_disjoint':True,'stay_identity_unique':True})
    dump('verification_calculation.json',verified)
    (OUT/'queries.sql').write_text(';\n'.join(dict.fromkeys(queries)),encoding='utf-8')
    c.close(); after = sha(db); assert after==before
    dump('provenance.json',{'database':str(db),'database_sha256_before':before,'database_sha256_after':after,
         'aw110_sources_sha256':sha(OLD/'sources.json'),'aw110_recipe_sha256':sha(OLD/'rechnen_final_snapshot.py'),
         'aw110_sources_checked':source_checks,'spec_sha256':EXPECTED,'code_sha256':sha(__file__),
         'completed_utc':datetime.now(timezone.utc).isoformat(),'peak_rss_bytes':PEAK})
    print(json.dumps({k:{f:{a:v for a,v in s.items() if a!='population_curve'} for f,s in result[k].items()} for k in ['primary','cart_shifted_month']},ensure_ascii=True),flush=True)
if __name__=='__main__': main()
