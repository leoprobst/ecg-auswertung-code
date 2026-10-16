import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from .common import *
import pickle,gc
from werkzeuge.varianten.adapter import release_import,prepare_record,context
PLANES={'frontal':(0,1,1,1),'transversal':(0,2,1,-1),'rechts_sagittal':(2,1,-1,1)}
def angle(a,b):
    if a is None or b is None or min(np.linalg.norm(a),np.linalg.norm(b))<1e-9:return np.nan
    return float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
def turn(a,plane,threshold=.15):
    if a is None or len(a)<4:return None
    i,j,sx,sy=PLANES[plane];xy=a[:,[i,j]]*np.array([sx,sy]);c=xy[:-1,0]*xy[1:,1]-xy[1:,0]*xy[:-1,1]
    pos=c[c>0].sum();neg=-c[c<0].sum();total=pos+neg
    if total<1e-6:return None
    net=pos-neg;major='tie' if abs(net)<1e-12 else ('cw' if net>0 else 'ccw');minor=float(min(pos,neg)/total)
    return {'net':float(net),'minor':minor,'majority':major,'class':'figure8' if minor>=threshold else major}
def measure(segments):
    out={};q=segments.get('qrs');t=segments.get('t')
    for mode in ['integral','peak']:
        def vec(a):return None if a is None else np.trapezoid(a,dx=2,axis=0) if mode=='integral' else a[np.argmax(np.linalg.norm(a,axis=1))]
        out[mode+'_angle']=angle(vec(q),vec(t))
    for s,a in segments.items():
        if s not in ['qrs','t']:continue
        v=np.trapezoid(a,dx=2,axis=0);out[s+'_area_vector']=v.tolist();out[s+'_area_norm']=float(np.linalg.norm(v))
        for plane in PLANES:out[s+'_'+plane]=turn(a,plane)
    return out
def ptb_package(r,neutral=False):
    import wfdb
    from vcg_biomarker.core.preprocessing import resample_signal,prepare_signal_paths,construct_robust_morphology_record
    from vcg_biomarker.mimic.signals import _convert_to_millivolts
    from vcg_biomarker.core.biomarkers.catalog_biomarkers import FeatureContext
    from vcg_biomarker.core.vcg_geometry import estimate_physiological_vcg_origin
    fr=release_import();path=Path(str(data_path("a.extern_ptbdb")))/r['record'];rawrecord=wfdb.rdrecord(str(path));names=[n.casefold() for n in rawrecord.sig_name]
    inds=[names.index(n.casefold()) for n in ('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6','vx','vy','vz')]
    raw=_convert_to_millivolts(rawrecord.p_signal[:,inds],tuple(rawrecord.units[j] or '' for j in inds));raw=resample_signal(raw,float(rawrecord.fs),500)
    paths=prepare_signal_paths(raw,500,powerline=50.);peaks=fr.detect_r_peaks_multilead(paths['delineation'][:,:12],500)
    xyz=fr.VCGTransformRegistry.transform(paths['delineation'][:,:12],'kors').xyz
    d=fr.FusionDelineator(weights={'neurokit':.4,'multilead':.35,'vcg':0. if neutral else .25},conflict_threshold=20.).delineate_record(ecg_12lead=paths['delineation'][:,:12],r_peaks=peaks,fs=500,ecg_id=0,vcg_xyz=None if neutral else xyz)
    robust,diagnostics=construct_robust_morphology_record(paths['morphology'],d,minimum_beats=3)
    if diagnostics['status']!='used':raise ValueError('median not used')
    ctx=FeatureContext(d,robust[:,:12],fr.VCGTransformRegistry.transform(robust[:,:12],'kors').xyz,500)
    if neutral:
        o=estimate_physiological_vcg_origin(ctx.vcg_xyz,d,500).primary
        if o is None:raise ValueError('origin unavailable')
        region=[o.start_sample,o.end_sample];bounds={s:ctx.bounds(s) if ctx.is_segment_evaluable(s) else None for s in ['qrs','t']}
    else:region=r['data']['origin_region'];bounds=r['data']['bounds']
    lo,hi=region;methods={m:fr.VCGTransformRegistry.transform(robust[:,:12],m).xyz for m in ['kors','inverse_dower']};methods['frank']=robust[:,12:15]
    centered={m:a-np.median(a[lo:hi+1],axis=0) for m,a in methods.items()}
    # Only Z/V4 changes; hold the minus origin region and transfer its V4 baseline linearly.
    v4=robust[:,9]-np.median(robust[lo:hi+1,9]);plus=centered['inverse_dower'].copy();plus[:,2]+=.126*v4;centered['plus']=plus
    segs={}
    for m,a in centered.items():
        segs[m]={}
        for s in ['qrs','t']:
            b=bounds.get(s)
            if b is None:continue
            x=a[b[0]:b[1]+1]
            # Fixed original availability for coefficient sensitivity; neutral uses amplitude gates anew.
            good=(s in r['data']['segments'].get(m if m!='plus' else 'inverse_dower',{})) if not neutral else np.max(np.linalg.norm(x,axis=1))>=(.1 if s=='qrs' else .03)
            if good:segs[m][s]=x
    errors=[]
    if not neutral:
        for m in ['kors','inverse_dower','frank']:
            for s,x in segs[m].items():errors.append(float(np.max(np.abs(x-r['data']['segments'][m][s]))))
        assert max(errors,default=0)<1e-8,'PTB original signal not reproduced'
    return segs,{'original_max_abs_error':max(errors,default=0),'neutral':neutral,'origin_region':region,'bounds':bounds}
def variant(row):
    package=prepare_record(row);ctx=context(package,'inverse_dower');a=ctx.origin_corrected_vcg();assessment=ctx.vcg_origin_assessment
    if a is None or assessment.primary is None:raise ValueError('Origin nicht schaetzbar')
    o=assessment.primary;baseline=float(np.median(package['paths']['morphology'][o.start_sample:o.end_sample+1,9]))
    plus=a.copy();plus[:,2]+=.126*(package['robust'][:,9]-baseline)
    segs={'inverse_dower':{},'plus':{}}
    for s in ['qrs','t']:
        bounds=ctx.bounds(s)
        if bounds is None or not ctx.is_segment_evaluable(s):continue
        lo,hi=bounds
        for m,x in [('inverse_dower',a),('plus',plus)]:segs[m][s]=x[lo:hi+1]
    return segs,{'origin_region':[o.start_sample,o.end_sample],'origin_status':assessment.status,'bounds':{s:ctx.bounds(s) for s in ['qrs','t']}}
def ptb_fast(r):
    """Exact coefficient correction from cached robust window + unchanged raw origin.

    Fallback whenever the origin touches a template region not in the cached window.
    No segmentation, baseline-region or availability change.
    """
    data=r['data'];qb=data['bounds'].get('qrs');ecg=data.get('ecg');region=data['origin_region']
    if qb is None or ecg is None:return ptb_package(r)
    grid=data['grid_ms'];start=int(qb[0]+grid[0]/2);end=int(qb[0]+grid[-1]/2)
    for s in ['qrs','t']:
        b=data['bounds'].get(s)
        if b is not None and not (start<=b[0]<=b[1]<=end):return ptb_package(r)
    lo,hi=region;diag=data['diagnostics'];targetstart=diag['target_start_sample'];targetend=diag['target_end_sample']
    if start<=lo<=hi<=end:baseline=float(np.median(ecg[lo-start:hi-start+1,9]));route='cached_robust_origin'
    elif hi<targetstart or lo>targetend:
        import wfdb
        from vcg_biomarker.core.preprocessing import resample_signal,prepare_signal_paths
        from vcg_biomarker.mimic.signals import _convert_to_millivolts
        path=Path(str(data_path("a.extern_ptbdb")))/r['record'];header=wfdb.rdheader(str(path));j=[n.casefold() for n in header.sig_name].index('v4')
        rawrecord=wfdb.rdrecord(str(path),channels=[j]);raw=_convert_to_millivolts(rawrecord.p_signal,tuple(rawrecord.units));raw=resample_signal(raw,float(rawrecord.fs),500)
        # Each preprocessing operation acts leadwise; no multilead delineation is invoked.
        v4=prepare_signal_paths(raw,500,powerline=50.)['morphology'][:,0]
        baseline=float(np.median(v4[lo:hi+1]));route='unaltered_origin_outside_template'
    else:return ptb_package(r)
    segs={m:{s:np.asarray(x) for s,x in data['segments'][m].items() if s in ['qrs','t']} for m in ['kors','inverse_dower','frank']};segs['plus']={}
    for s,a in segs['inverse_dower'].items():
        b=data['bounds'][s];v4=ecg[b[0]-start:b[1]-start+1,9]
        if len(v4)!=len(a) or not np.isfinite(v4).all():return ptb_package(r)
        plus=a.copy();plus[:,2]+=.126*(v4-baseline);segs['plus'][s]=plus
    return segs,{'original_max_abs_error':0.,'origin_region':region,'bounds':data['bounds'],'fastpath':route,'negative_source':'cached_original_segments_preserved'}
def compute_variant(item):
    i,row=item;guard('sign_variant_'+str(i));tic=time.time()
    result={'population':'Varianten','subject':row['subject_id'],'study_id':row['study_id'],'fold':row['split']}
    try:segs,audit=variant(row);result.update(status='ok',methods={m:measure(v) for m,v in segs.items()},audit=audit)
    except ValueError as e:result.update(status='abstention',error=str(e))
    result['seconds']=time.time()-tic
    return i,result
def scientific_digest(result):
    selected={k:v for k,v in result.items() if k!='seconds'}
    return hashlib.sha256(json.dumps(selected,sort_keys=True,default=str).encode()).hexdigest()
def run():
    release_import();dest=OUT/'sign';dest.mkdir(exist_ok=True)
    src=Path(str(data_path("a.kompendium_musterseite_v2_frank_processed_privat_pkl")))
    with src.open('rb') as f:records=[r for r in pickle.load(f) if r['primary']]
    inputs=[src,src.with_name('freeze.json')];start=time.time()
    # Validate the fast path against all already independently recomputed full checkpoints.
    validation_path=OUT/'sign_fastpath_validation_privat.json'
    fastchecks=json.loads(validation_path.read_text()) if validation_path.exists() else []
    assert not fastchecks or (len(fastchecks)>=15 and max(r['max_abs_error'] for r in fastchecks)<1e-8)
    # Resume an unchanged correction; do not repeat completed full signal work.
    previous_inputs=json.loads((MA/'frank_inputs.json').read_text())
    assert sha(src)==previous_inputs['files'][str(src)]['sha256']
    todo=[] if fastchecks else records
    for r in todo:
        p=dest/('ptb_'+r['patient']+'.json')
        if not p.exists():continue
        old=json.loads(p.read_text());segs,audit=ptb_fast(r);new={m:measure(v) for m,v in segs.items()}
        if old['status']!='ok':continue
        errors=[]
        for m in ['kors','inverse_dower','plus','frank']:
            for key,value in new[m].items():
                ref=old['methods'][m].get(key)
                if isinstance(value,(float,int)):errors.append(abs(value-ref) if np.isfinite(value) and np.isfinite(ref) else 0. if np.isnan(value) and np.isnan(ref) else float('inf'))
                elif isinstance(value,list):errors.append(float(np.max(abs(np.array(value)-np.array(ref)))))
                elif isinstance(value,dict):assert value==ref
        assert max(errors,default=0)<1e-8,'Fast correction differs from full recomputation'
        fastchecks.append({'patient':r['patient'],'max_abs_error':max(errors,default=0),'route':audit.get('fastpath','full_fallback')})
    if not validation_path.exists():dump(validation_path,fastchecks)
    for i,r in enumerate(records):
        guard('sign_ptb_'+str(i));p=dest/('ptb_'+r['patient']+'.json')
        if p.exists():continue
        try:segs,audit=ptb_fast(r);result={'population':'PTB','subject':r['patient'],'fold':'extern_technisch','group':r['group'],'status':'ok','methods':{m:measure(v) for m,v in segs.items()},'audit':audit}
        except Exception as e:result={'population':'PTB','subject':r['patient'],'fold':'extern_technisch','status':'failed','error':str(e)}
        dump(p,result)
        if i%25==0:print('SIGN_PTB',i+1,len(records),flush=True)
    from werkzeuge.transform_katalog.fortsetzung import rows as frozen_rows
    sample=frozen_rows();assert len(sample)==5648
    # One Python process, two case workers; numerical library threads remain one.
    # FeatureContext is explicitly threading.local in the bound release.
    from concurrent.futures import ThreadPoolExecutor
    checks=[]
    candidates=[(i,row) for i,row in enumerate(sample) if (dest/f'variant_{i:04d}.json').exists()][:32]
    assert len(candidates)==32
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i,new in executor.map(compute_variant,candidates):
            old=json.loads((dest/f'variant_{i:04d}.json').read_text());assert scientific_digest(new)==scientific_digest(old),'Two-worker scientific mismatch'
            checks.append({'index':i,'scientific_sha256':scientific_digest(new),'same_as_serial':True})
    dump(OUT/'sign_worker_validation.json',{'python_processes':1,'case_workers':2,'numeric_threads':1,'release_context':'threading.local','checks':checks,'comparison':'identical status, methods, origin and bounds; elapsed time excluded'})
    remaining=[]
    for i,row in enumerate(sample):
        p=dest/f'variant_{i:04d}.json'
        if p.exists():assert json.loads(p.read_text())['study_id']==row['study_id']
        else:remaining.append((i,row))
    with ThreadPoolExecutor(max_workers=2) as executor:
        for count,(i,result) in enumerate(executor.map(compute_variant,remaining),1):
            dump(dest/f'variant_{i:04d}.json',result)
            if count%20==0:gc.collect()
            if count%50==0:print('SIGN_VARIANT',i+1,len(sample),round(time.time()-start),flush=True)
    from werkzeuge.transform_katalog.inventar import SAMPLE,CONTEXT,FREEZE
    inputs+=[SAMPLE,CONTEXT,FREEZE]
    # Individual raw binding from frozen waveform headers and actual data filenames.
    rawmanifest=[]
    import wfdb
    root=Path(str(data_path("a.extern_ptbdb")));checksums={line.split(maxsplit=1)[1].strip().lstrip('*'):line.split()[0] for line in (root/'SHA256SUMS.txt').read_text().splitlines() if line.strip()}
    for r in records:
        base=root/r['record'];header=wfdb.rdheader(str(base));paths=[Path(str(base)+'.hea')]+[base.parent/name for name in set(header.file_name)]
        for p in paths:
            h=sha(p);assert h==checksums[p.relative_to(root).as_posix()],'PTB raw hash mismatch'
            rawmanifest.append({'path':str(p),'sha256':h})
    for row in sample:
        base=Path(row['waveform_path']);base=base.with_suffix('') if base.suffix in ('.hea','.dat') else base
        for ext in ['.hea','.dat']:
            p=Path(str(base)+ext)
            if p.exists():rawmanifest.append({'path':str(p),'sha256':sha(p)})
    dump(OUT/'sign_raw_hashes_privat.json',rawmanifest)
    aggregate();bind('sign',inputs+[OUT/'sign_raw_hashes_privat.json'],[OUT/'sign_continuous.csv',OUT/'sign_class_changes.csv',OUT/'sign_summary.json'])
def aggregate():
    records=[json.loads(p.read_text()) for p in sorted((OUT/'sign').glob('*.json'))];continuous=[];changes=[]
    for r in records:
        if r['status']!='ok':continue
        minus=r['methods']['inverse_dower'];plus=r['methods']['plus']
        base={k:r[k] for k in ['population','fold']};base['subject']=str(r['subject'])
        for key in ['integral_angle','peak_angle','qrs_area_norm','t_area_norm']:
            a=minus.get(key,np.nan);b=plus.get(key,np.nan)
            continuous.append({**base,'metric':key,'minus':a,'plus':b,'delta':b-a})
        for s in ['qrs','t']:
            v1=minus.get(s+'_area_vector');v2=plus.get(s+'_area_vector')
            if v1 is not None and v2 is not None:
                continuous.append({**base,'metric':s+'_area_direction_deg','minus':0.,'plus':angle(np.array(v1),np.array(v2)),'delta':angle(np.array(v1),np.array(v2))})
                for j,ax in enumerate(['X','Y','Z']):continuous.append({**base,'metric':s+'_area_'+ax,'minus':v1[j],'plus':v2[j],'delta':v2[j]-v1[j]})
            for plane in PLANES:
                a=minus.get(s+'_'+plane);b=plus.get(s+'_'+plane)
                if a is None or b is None:continue
                for th in ['net',.05,.1,.15,.25]:
                    ca=a['majority'] if th=='net' else 'figure8' if a['minor']>=th else a['majority'];cb=b['majority'] if th=='net' else 'figure8' if b['minor']>=th else b['majority']
                    changes.append({**base,'segment':s,'plane':plane,'threshold':str(th),'minus_class':ca,'plus_class':cb,'switch':int(ca!=cb)})
        if r['population']=='PTB':
            frank=r['methods']['frank']
            for m in ['inverse_dower','plus']:
                for key in ['integral_angle','peak_angle']:
                    a=frank.get(key,np.nan);b=r['methods'][m].get(key,np.nan)
                    continuous.append({**base,'metric':m+'_minus_Frank_'+key,'minus':a,'plus':b,'delta':b-a})
    cf=pd.DataFrame(continuous);sf=pd.DataFrame(changes);cf.to_parquet(OUT/'sign_continuous_privat.parquet',index=False);sf.to_parquet(OUT/'sign_changes_privat.parquet',index=False)
    out=[]
    for keys,z in cf.groupby(['population','fold','metric']):
        v=z.delta.to_numpy(float);v=v[np.isfinite(v)];n=len(v)
        if not n:continue
        rng=np.random.default_rng(SEED);ix=rng.integers(0,n,(B,n));bb=v[ix]
        lo,hi=ci(bb.mean(1));out.append({'population':keys[0],'fold':keys[1],'metric':keys[2],'n_pairs':n,'n_subjects':z.subject.nunique(),'mean_delta':v.mean(),'median_delta':np.median(v),'p95_abs_delta':np.quantile(abs(v),.95),'mean_ci_low':lo,'mean_ci_high':hi})
    pd.DataFrame(out).to_csv(OUT/'sign_continuous.csv',index=False)
    out=[]
    for keys,z in sf.groupby(['population','fold','segment','plane','threshold']):
        v=z.switch.to_numpy();n=len(v);rng=np.random.default_rng(SEED);ix=rng.integers(0,n,(B,n));lo,hi=ci(v[ix].mean(1))
        out.append({'population':keys[0],'fold':keys[1],'segment':keys[2],'plane':keys[3],'threshold':keys[4],'n_pairs':n,'n_subjects':z.subject.nunique(),'switch_count':int(v.sum()),'fraction':v.mean(),'ci_low':lo,'ci_high':hi})
    pd.DataFrame(out).to_csv(OUT/'sign_class_changes.csv',index=False)
    frontal=sf[sf.plane.eq('frontal')];assert frontal.switch.sum()==0
    xy=cf[cf.metric.str.endswith('_X')|cf.metric.str.endswith('_Y')];assert np.nanmax(abs(xy.delta))==0
    fastchecks=json.loads((OUT/'sign_fastpath_validation_privat.json').read_text());full=[r for r in records if r['population']=='PTB' and r['status']=='ok' and 'fastpath' not in r['audit']]
    dump(OUT/'sign_summary.json',{'candidate_counts':pd.Series([r['population'] for r in records]).value_counts().to_dict(),'status_counts':{pop:pd.Series([r['status'] for r in records if r['population']==pop]).value_counts().to_dict() for pop in ['PTB','Varianten']},'frontal_switches':int(frontal.switch.sum()),'xy_area_max_abs_delta':float(np.nanmax(abs(xy.delta))),'coefficient_delta':.126,'same_original_availability':True,'PTB_max_original_error':max((r['audit']['original_max_abs_error'] for r in full),default=None),'PTB_independent_full_recomputations':len(full),'PTB_cached_preserved':sum(r['population']=='PTB' and r['status']=='ok' and 'fastpath' in r['audit'] for r in records),'fastpath_full_comparisons':len(fastchecks),'fastpath_max_error':max((r['max_abs_error'] for r in fastchecks),default=None)})
def neutral():
    release_import();src=Path(str(data_path("a.kompendium_musterseite_v2_frank_processed_privat_pkl")))
    with src.open('rb') as f:records=[r for r in pickle.load(f) if r['primary']]
    selected=sorted(records,key=lambda r:hashlib.sha256(('AW077-neutral:'+r['patient']).encode()).hexdigest())[:50]
    dump(OUT/'neutral_selection_privat.json',[{k:r[k] for k in ['patient','record']} for r in selected]);rows=[]
    for i,r in enumerate(selected):
        guard('neutral_'+str(i))
        try:
            segs,audit=ptb_package(r,True);m={k:measure(v) for k,v in segs.items()}
            for method in ['kors','inverse_dower']:
                for mode in ['integral','peak']:
                    key=mode+'_angle';rows.append({'subject':r['patient'],'method':method,'mode':mode,'new_delta':m[method][key]-m['frank'][key],'old_delta':measure(r['data']['segments'][method])[key]-measure(r['data']['segments']['frank'])[key]})
        except ValueError as e:rows.append({'subject':r['patient'],'error':str(e)})
    f=pd.DataFrame(rows);f.to_csv(OUT/'neutral_detail_privat.csv',index=False);out=[]
    for keys,z in f.dropna(subset=['method']).groupby(['method','mode']):
        good=np.isfinite(z.old_delta)&np.isfinite(z.new_delta);z=z[good];x=z.new_delta.to_numpy();d=(z.new_delta-z.old_delta).to_numpy();rng=np.random.default_rng(SEED);ix=rng.integers(0,len(x),(B,len(x)))
        out.append({'method':keys[0],'mode':keys[1],'n_selected':50,'n_same_cases':len(x),'old_bias':float(z.old_delta.mean()),'neutral_bias':float(x.mean()),'neutral_p95_abs':float(np.quantile(abs(x),.95)),'bias_difference':float(d.mean()),'bias_difference_ci_low':ci(d[ix].mean(1))[0],'bias_difference_ci_high':ci(d[ix].mean(1))[1]})
    pd.DataFrame(out).to_csv(OUT/'neutral_summary.csv',index=False);bind('neutral',[src,src.with_name('freeze.json')],[OUT/'neutral_selection_privat.json',OUT/'neutral_summary.csv'])
if __name__=='__main__':neutral() if 'neutral' in sys.argv else run()
