"""One-time W/M/P runner of the test-partition replication with immutable statistical checkpoints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
from repo_config import check_reference
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,gc,traceback
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from scipy.stats import spearmanr,rankdata
from . import aw163_extract as e
from .aw163_preflight import ROOT,OUT,REL,sha,EXCLUDED,PRELOADED
from .aw163_statistics import ci,weights,weighted_metrics,combine,agreement,selftest
from . import torlogik as gates
OLD=e.OLD
NAMES=e.readsmall(OUT/'marker_universes.json')['catalog']
U=e.readsmall(OUT/'marker_universes.json')
CODE_NAMES=['stabil','bedingt','transformabhaengig','nicht_schaetzbar','konstant_identisch']
def read(p,cols):
    e.log('begin',type='marker_projection',path=str(p),columns=list(cols),freeze=sha(OUT/'statistical_freeze.json'))
    f=e.a.readp(p,cols,test=True)
    e.log('end',type='marker_projection',path=str(p),rows=len(f))
    return f
def checkpoint(n,x):
    p=OUT/n
    if p.exists():
        assert e.readsmall(p)==e.clean(x),'Checkpoint content differs: '+n
        return
    e.a.save(n,e.clean(x))
def freeze():
    assert not (OUT/'statistical_freeze.json').exists()
    e.public('Start of input and statistics binding','Integrity and structure only; before any new test statistic')
    assert e.readsmall(OUT/'development_statistics_check.json')['passed']
    for kind in ('W','M'):assert e.readsmall(OUT/f'status_{kind}finalize.json')['state']=='complete'
    creation={}
    with (OUT/'access_private.jsonl').open(encoding='utf8') as ledger:
        for line in ledger:
            item=json.loads(line)
            if item.get('event')=='end' and item.get('type')=='waveform' and item.get('checkpoint_sha256'):
                creation[(item['phase'][0],Path(item['path']).stem)]=item['checkpoint_sha256']
    checked={}
    for kind in ('W','M'):
        count=0
        for item in e.readsmall(OUT/f'{kind}_input_binding.json')['checkpoints']:
            p=Path(item['path']);expected=creation[(kind,p.stem)]
            e.log('begin',type='checkpoint_integrity',path=str(p),expected_sha256=expected)
            assert sha(p)==expected==item['sha256'],'Own extraction checkpoint changed'
            e.log('end',type='checkpoint_integrity',path=str(p),sha256=expected);count+=1
        checked[kind]=count
    checkpoint('extraction_checkpoint_integrity.json',{'passed':True,'compared_to_original_creation_ledger':True,'checked':checked})
    checkpoint('selftests_statistics.json',selftest())
    inputs={str(OUT/n):sha(OUT/n) for n in ['extraction_freeze.json','continuation_binding_v2.json','W_input_binding.json','M_input_binding.json','W_features_private.parquet','M_features_private.parquet','development_benchmarks.json','development_W_gesamt.csv','development_W_proxy.csv','development_M.csv','marker_universes.json','identity_gate.json','development_statistics_check.json','selftests_pre_statistics.json','extraction_checkpoint_integrity.json']}
    first=e.readsmall(OUT/'extraction_freeze.json')
    for item in first['inputs']:
        assert sha(item['path'])==item['sha256'];inputs[item['path']]=item['sha256']
    for p in [ROOT/'publikationen/PUB-01_messvaliditaet/supplementary_data_S2_marker_reliability.csv',ROOT/'abbildungen/v11/AW149_cluster_membership.csv',ROOT/'werkzeuge/rv_a1/wiederholbarkeit.py',ROOT/'werkzeuge/messaudit/statistik.py',ROOT/'werkzeuge/rv_a1/vorzeichen.py']:
        inputs[str(p)]=sha(p)
    for p in [Path(str(data_path("a.rv_a1_repeat_summary_csv"))),Path(str(data_path("a.rv_a1_p_duration_csv")))]:inputs[str(p)]=sha(p)
    manifest={'version':'AW163-v2','base_manifest_sha256':first['manifest_v1_sha256'],'inputs':inputs,'W':{'path':str(OUT/'W_features_private.parquet'),'binding_sha256':sha(OUT/'W_input_binding.json')},'M':{'path':str(OUT/'M_features_private.parquet'),'binding_sha256':sha(OUT/'M_input_binding.json')},'P':next(x for x in first['inputs'] if x['role']=='P_release_or_cache_values'),'release_implementation_hash':first['release_implementation_hash'],'technical_excluded':sorted(EXCLUDED),'M_preloaded':sorted(PRELOADED),'new_test_statistics_computed':False}
    checkpoint('input_manifest_v2.json',manifest)
    code={str(p):sha(p) for p in sorted((ROOT/'werkzeuge/a_testreplikation').glob('aw163_*')) if p.is_file()}
    code[str(ROOT/'werkzeuge/a_testreplikation/torlogik.py')]=sha(ROOT/'werkzeuge/a_testreplikation/torlogik.py')
    code[str(ROOT/'werkzeuge/a_testreplikation/aw159_inputs.py')]=sha(ROOT/'werkzeuge/a_testreplikation/aw159_inputs.py')
    for item in e.readsmall(OUT/'source_hash_checks.json'):
        assert sha(item['path'])==item['expected'],'Bound definition source changed'
        inputs[item['path']]=item['expected']
    for item in e.readsmall(OLD/'release_python_manifest.json'):
        assert sha(item['path'])==item['sha256'],'Release source changed'
        code[item['path']]=item['sha256']
    checkpoint('statistical_freeze.json',{'utc':e.a.utc(),'code':code,'inputs':inputs,'manifest_v2_sha256':sha(OUT/'input_manifest_v2.json'),'seeds':{'W':1461201,'M0':1461202,'M1':1461203,'P':1461204},'bootstrap':1000,'chunk':20,'decision_rules':'specification v1.0 plus Nachtrag1; no changes','selftests_sha256':sha(OUT/'selftests_statistics.json')})
    e.public('Manifest v2 and binding before test computation',f'Manifest `{sha(OUT/"input_manifest_v2.json")}`; statistics freeze `{sha(OUT/"statistical_freeze.json")}`; no new test statistics yet')
def w_analysis():
    source=OUT/'W_features_private.parquet'
    z=read(OLD/'w_candidates_private.parquet',pq.ParquetFile(OLD/'w_candidates_private.parquet').schema_arrow.names)
    ids=np.sort(z.subject_id.unique());checkpoint('W_bootstrap_population_private.json',ids.tolist())
    status=read(source,['subject_id','study_id','split','extraction_status','signal_sha256','signal_quality_score','context_heart_rate_bpm']).set_index('study_id')
    check_reference('aw163_run.status_rows',len(status))
    assert not status.index.duplicated().any()
    first=status.reindex(z.first_study_id);second=status.reindex(z.second_study_id)
    assert np.array_equal(first.subject_id.to_numpy(),z.subject_id.to_numpy()) and np.array_equal(second.subject_id.to_numpy(),z.subject_id.to_numpy())
    rawby={int(Path(v['path']).stem):{Path(k).suffix:h for k,h in v['raw_sha256'].items()} for v in e.readsmall(OUT/'W_input_binding.json')['checkpoints']}
    simultaneous_before=z.hours_between.eq(0).to_numpy()
    raw_header_equal=np.array([rawby[int(a)]['.hea']==rawby[int(b)]['.hea'] for a,b in zip(z.first_study_id,z.second_study_id)])
    raw_data_equal=np.array([rawby[int(a)]['.dat']==rawby[int(b)]['.dat'] for a,b in zip(z.first_study_id,z.second_study_id)])
    checkpoint('W_simultaneous_before_gates.json',{'candidate_simultaneous_pairs':int(simultaneous_before.sum()),'equal_header_sha256':int((simultaneous_before&raw_header_equal).sum()),'equal_data_sha256':int((simultaneous_before&raw_data_equal).sum()),'equal_header_and_data_sha256':int((simultaneous_before&raw_header_equal&raw_data_equal).sum()),'bound_raw_headers_and_data_for_all_candidates':True})
    identical=first.signal_sha256.notna().to_numpy()&second.signal_sha256.notna().to_numpy()&(first.signal_sha256.to_numpy()==second.signal_sha256.to_numpy())
    qc=(first.extraction_status.eq('ok')&first.signal_quality_score.fillna(0).ge(.6)).to_numpy()&(second.extraction_status.eq('ok')&second.signal_quality_score.fillna(0).ge(.6)).to_numpy()
    checkpoint('W_pair_gate.json',{'candidates':len(z),'persons_before_QC':len(ids),'identical_signal_hash_removed':int(identical.sum()),'QC_failed_after_identity':int((~identical&~qc).sum()),'pairs_after_QC':int((~identical&qc).sum()),'time_equal_before_QC':int(z.hours_between.eq(0).sum()),'missing_hashes_before_QC':int((first.signal_sha256.isna().to_numpy()|second.signal_sha256.isna().to_numpy()).sum()),'intervening_intervention_checked':True,'source':'AW159 W selection and newly extracted release status/hash/QC'})
    z=z.loc[~identical&qc].reset_index(drop=True)
    cols=['subject_id','study_id','hadm_id','admission_match_ambiguous','machine_metadata_usable_for_primary','machine_report','machine_qrs_duration_ms','machine_cart_id','ecg_time']
    master=Path(str(data_path("a.mimic_validation_master_ecg_master_index_parquet")))
    assert sha(master)==e.readsmall(OLD/'disjointness_attestation.json')['master_sha256']
    import pyarrow.dataset as ds
    e.log('begin',type='context',path=str(master),columns=cols,filter='selected W studies; private')
    pq.ParquetFile(master);m=ds.dataset(str(master),format='parquet').to_table(columns=cols,filter=ds.field('study_id').isin(status.index.tolist())).to_pandas()
    e.log('end',type='context',path=str(master),rows=len(m))
    z['signal_quality_first']=status.reindex(z.first_study_id).signal_quality_score.to_numpy();z['signal_quality_second']=status.reindex(z.second_study_id).signal_quality_score.to_numpy()
    env={'np':np,'pd':pd};e.a.ast_functions(ROOT/'werkzeuge/messaudit/statistik.py',['subgroup_frame'],env);z=env['subgroup_frame'](z,m)
    proxyids=np.sort(z.loc[z.stable,'subject_id'].unique());order=z.sort_values(['subject_id','first_ecg_time','second_ecg_time','first_study_id','second_study_id'])
    one=np.zeros(len(z),bool);one[order.drop_duplicates('subject_id').index]=True
    pone=np.zeros(len(z),bool);pone[order.loc[order.stable].drop_duplicates('subject_id').index]=True
    masks={'gesamt':np.ones(len(z),bool),'proxy':z.stable.to_numpy(),'ohne_zeitgleich':z.hours_between.ne(0).to_numpy(),'gesamt_proxy_personen':z.subject_id.isin(proxyids).to_numpy(),'gesamt_ein_paar':one,'proxy_ein_paar':pone,'proxy_quality08':z.stable_quality.to_numpy()}
    checkpoint('W_denominators.json',{g:{'pairs':int(mask.sum()),'persons':int(z.loc[mask,'subject_id'].nunique())} for g,mask in masks.items()})
    delta=np.abs(status.reindex(z.second_study_id).context_heart_rate_bpm.to_numpy(float)-status.reindex(z.first_study_id).context_heart_rate_bpm.to_numpy(float))
    hr=[]
    for g,mask in masks.items():
        v=delta[mask];v=v[np.isfinite(v)]
        hr.append({'population':g,'candidate_pairs':int(mask.sum()),'finite_pairs':len(v),'missing_pairs':int(mask.sum())-len(v),'median_abs_delta_bpm':float(np.median(v)) if len(v) else None,'p95_abs_delta_bpm':float(np.quantile(v,.95)) if len(v) else None,'fraction_gt5':float(np.mean(v>5)) if len(v) else None,'fraction_gt10':float(np.mean(v>10)) if len(v) else None,'fraction_gt20':float(np.mean(v>20)) if len(v) else None,'source_field':'release context_heart_rate_bpm; no empirical alias'})
    pd.DataFrame(hr).to_csv(OUT/'W_heart_rate_delta.csv',index=False)
    pairpath=OUT/'W_pairs_after_gates_private.parquet'
    if not pairpath.exists():e.a.writep('W_pairs_after_gates_private.parquet',z)
    checkpoint('W_pair_selection_binding.json',{'sha256':sha(pairpath),'freeze':sha(OUT/'statistical_freeze.json'),'rows':len(z)})
    # Retained simultaneous pairs: nonmissing unequal release signal hashes resolve identity.
    af=status.reindex(z.first_study_id);bf=status.reindex(z.second_study_id);sim=z.hours_between.eq(0).to_numpy()
    checkpoint('W_simultaneous_identity.json',{'retained_simultaneous_pairs':int(sim.sum()),'unequal_nonmissing_decoded_signal_sha256':int((sim&af.signal_sha256.notna().to_numpy()&bf.signal_sha256.notna().to_numpy()&(af.signal_sha256.to_numpy()!=bf.signal_sha256.to_numpy())).sum()),'unresolved':int((sim&(af.signal_sha256.isna().to_numpy()|bf.signal_sha256.isna().to_numpy())).sum()),'raw_header_data_sha256_in_extraction_checkpoints':True})
    env={'np':np,'B':1000};e.a.ast_functions(ROOT/'werkzeuge/rv_a1/wiederholbarkeit.py',['estimate','moments'],env)
    rows=[];boots={g:[] for g in masks}
    for block in range(0,len(NAMES),16):
        e.a.guard();names=NAMES[block:block+16]
        p=OUT/'W_statistics_checkpoints';p.mkdir(exist_ok=True)
        dest=p/f'block_{block:04d}.npz';rowpath=p/f'block_{block:04d}.csv';bindpath=p/f'block_{block:04d}_binding.json'
        if bindpath.exists():
            bound=e.readsmall(bindpath);assert bound['freeze']==sha(OUT/'statistical_freeze.json') and bound['names']==names
            assert sha(dest)==bound['output_sha256'] and sha(rowpath)==bound['row_sha256']
            saved=np.load(dest)
            for g in masks:boots[g].append(saved[g][:,:,0])
            rows.extend(pd.read_csv(rowpath,float_precision='round_trip').to_dict('records'));saved.close();continue
        blockrows=[]
        f=read(source,['study_id',*names]).set_index('study_id');x=f.reindex(z.first_study_id).to_numpy(float);y=f.reindex(z.second_study_id).to_numpy(float)
        saved={}
        for g,mask in masks.items():
            pp,_,nn,ns,bsd,den,scale=env['moments'](x[mask],y[mask],z.loc[mask,'subject_id'].to_numpy(),ids,None)
            xx=x[mask];yy=y[mask];ok=np.isfinite(xx)&np.isfinite(yy)
            center=np.nanmean(np.where(ok,(xx+yy)/2,np.nan),axis=0)
            a=np.where(ok,(xx-center)/scale,0);b=np.where(ok,(yy-center)/scale,0)
            iv=np.searchsorted(ids,z.loc[mask,'subject_id'].to_numpy());agg=[]
            for v in [ok.astype(float),a,b,a*a,b*b,a*b]:
                vagg=np.zeros((len(ids),len(names)));np.add.at(vagg,iv,v);agg.append(vagg)
            boot=np.full((1000,len(names),4),np.nan)
            for lo,w in weights(ids,1461201):
                bb,_=env['estimate']([w@v for v in agg]);effective=(w>0).astype(float)@(agg[0]>0)
                bb[effective<30]=np.nan;bb[:,:,1:]*=scale[None,:,None];boot[lo:lo+20]=bb
            boots[g].append(boot[:,:,0]);saved[g]=boot
            for j,n in enumerate(names):
                r={'population':g,'marker_id':n,'n_pairs':int(nn[j]),'n_subjects':int(ns[j]),'icc':pp[j,0],'bias':pp[j,1],'difference_sd':pp[j,2],'rc':pp[j,3],'sdc95':pp[j,3],'sem':pp[j,2]/np.sqrt(2),'between_person_sd':bsd[j]}
                for k,key in enumerate(('icc','bias','difference_sd','rc')):
                    c=ci(boot[:,j,k]);r.update({key+'_ci_low':c['low'],key+'_ci_high':c['high'],key+'_bootstrap_valid':c['valid']})
                for alias,key,factor in [('sdc95','rc',1.),('sem','difference_sd',1/np.sqrt(2))]:
                    for side in ('low','high'):
                        value=r[key+'_ci_'+side];r[alias+'_ci_'+side]=None if value is None else value*factor
                    r[alias+'_bootstrap_valid']=r[key+'_bootstrap_valid']
                blockrows.append(r)
        rows.extend(blockrows)
        assert not dest.exists() and not rowpath.exists();np.savez_compressed(dest,names=np.array(names),**saved);pd.DataFrame(blockrows).to_csv(rowpath,index=False)
        checkpoint(f'W_statistics_checkpoints/block_{block:04d}_binding.json',{'freeze':sha(OUT/'statistical_freeze.json'),'names':names,'seed':1461201,'output_sha256':sha(dest),'row_sha256':sha(rowpath)})
        e.a.save('progress_statistics.json',{'phase':'W','markers_completed':block+len(names),'total':len(NAMES)})
        gc.collect()
    frame=pd.DataFrame(rows)
    for g in ('gesamt','proxy'):
        dev=pd.read_csv(OUT/f'development_W_{g}.csv').set_index('marker_id')
        ix=frame.population.eq(g)
        aligned=dev.reindex(frame.loc[ix,'marker_id'])
        frame.loc[ix,'development_icc']=aligned.icc.to_numpy()
        frame.loc[ix,'RC_ratio_to_development']=frame.loc[ix,'rc'].to_numpy()/aligned.rc.to_numpy()
        frame.loc[ix,'bias_shift_over_development_SD']=np.abs(frame.loc[ix,'bias'].to_numpy()-aligned.bias.to_numpy())/aligned.between_person_sd.to_numpy()
        for field,ids,lo,hi in [('RC_ratio_to_development',U[f'W_RC_{g}'],.8,1.25),('bias_shift_over_development_SD',U[f'W_bias_{g}'],0,.25)]:
            v=frame.loc[ix,field].to_numpy();member=frame.loc[ix,'marker_id'].isin(ids).to_numpy()
            frame.loc[ix,field+'_criterion_status']=np.where(member,np.where(np.isfinite(v),np.where((v>=lo)&(v<=hi),'Kriterium_erfuellt','nicht_erfuellt'),'nicht_schaetzbar'),'ausserhalb_festem_Tornenner')
    frame['marker_anderweitig_testgemessen']=frame.marker_id.isin(PRELOADED)
    frame['F22_historical_provenance_status']='offen; betrifft insbesondere W-P, kein vollverblindeter W-Gesamtclaim'
    frame['technical_status']=np.where(np.isfinite(frame.icc),'schaetzbar','nicht_schaetzbar')
    summary=[];family=[];metadata=pd.read_csv(ROOT/'publikationen/PUB-01_messvaliditaet/supplementary_data_S2_marker_reliability.csv').set_index('marker_id')
    clusters=pd.read_csv(ROOT/'abbildungen/v11/AW149_cluster_membership.csv').set_index('marker_id')
    benchmark=e.readsmall(OUT/'development_benchmarks.json')['summaries']
    for g in masks:
        point=frame.loc[frame.population.eq(g)].set_index('marker_id').reindex(NAMES);bb=np.concatenate(boots[g],axis=1);a=point.icc.to_numpy()
        finite=np.isfinite(a);s={'population':g,'universe':len(NAMES),'estimable':int(finite.sum()),'median_icc':float(np.nanmedian(a)),'median_ci':ci(np.nanmedian(bb,axis=1)),'fraction_ge075_universe':float(np.sum(a>=.75)/len(NAMES)),'fraction_ge075_estimable':float(np.sum(a>=.75)/finite.sum()) if finite.any() else None,'fraction_ge075_universe_ci':ci((bb>=.75).sum(1)/len(NAMES)),'fraction_ge075_estimable_ci':ci((bb>=.75).sum(1)/np.isfinite(bb).sum(1)),'estimable_bootstrap_min':int(np.isfinite(bb).sum(1).min()),'estimable_bootstrap_max':int(np.isfinite(bb).sum(1).max()),'estimable_bootstrap_median':float(np.median(np.isfinite(bb).sum(1)))}
        if g in ('gesamt','proxy'):
            dev=pd.read_csv(OUT/f'development_W_{g}.csv').set_index('marker_id').reindex(NAMES)
            d=U[f'D_W_{g}'];rcids=U[f'W_RC_{g}'];biasids=U[f'W_bias_{g}'];valid=np.isfinite(a)&np.isfinite(dev.icc.to_numpy())
            rho=float(spearmanr(a[valid],dev.icc.to_numpy()[valid]).statistic) if valid.sum()>1 else np.nan
            brho=[]
            for b in bb:
                good=np.isfinite(b)&np.isfinite(dev.icc.to_numpy());brho.append(spearmanr(b[good],dev.icc.to_numpy()[good]).statistic if good.sum()>1 else np.nan)
            rc=point.loc[rcids].rc.to_numpy()/dev.loc[rcids].rc.to_numpy();bc=np.abs(point.loc[biasids].bias.to_numpy()-dev.loc[biasids].bias.to_numpy())/dev.loc[biasids].between_person_sd.to_numpy()
            s.update(development_median=benchmark[g]['median_icc'],coverage_denominator=len(d),coverage_numerator=int(np.isfinite(point.loc[d].icc).sum()),coverage=float(np.isfinite(point.loc[d].icc).mean()),rho=rho,rho_ci=ci(brho),rank_pairs=int(valid.sum()),rc_denominator=len(rcids),rc_numerator=int(((rc>=.8)&(rc<=1.25)).sum()),rc_fraction=float(((rc>=.8)&(rc<=1.25)).mean()),bias_denominator=len(biasids),bias_numerator=int((bc<=.25).sum()),bias_fraction=float((bc<=.25).mean()),original_denominator_sensitivity={'universe':2005,'coverage_denominator':len(d)+2,'coverage':float(np.isfinite(point.loc[d].icc).sum()/(len(d)+2)),'median_ci':s['median_ci'],'development_median_original':.4665168480877628 if g=='gesamt' else .5242050093521649})
            pos=dev.between_person_sd.gt(0)&np.isfinite(dev.between_person_sd)
            s['median_RC_over_development_SD']=float(np.nanmedian(point.loc[pos].rc/dev.loc[pos].between_person_sd));s['median_abs_bias_over_development_SD']=float(np.nanmedian(point.loc[pos].bias.abs()/dev.loc[pos].between_person_sd))
            di=[NAMES.index(n) for n in d];s['same_fixed_development_ICC_ids_sensitivity']={'IDs':len(d),'finite_test':int(np.isfinite(a[di]).sum()),'test_median':float(np.nanmedian(a[di])),'test_median_ci':ci(np.nanmedian(bb[:,di],axis=1)),'development_median':s['development_median']}
        for fam in sorted(metadata.family.unique()):
            chosen=[n for n in NAMES if metadata.loc[n,'family']==fam];v=point.loc[chosen]
            family.append({'population':g,'family':fam,'markers':len(chosen),'finite_icc':int(np.isfinite(v.icc).sum()),'median_icc':float(np.nanmedian(v.icc)) if np.isfinite(v.icc).any() else None})
        for unit in sorted(metadata.unit.dropna().unique()):
            chosen=[n for n in NAMES if metadata.loc[n,'unit']==unit];v=point.loc[chosen]
            family.append({'population':g,'unit':unit,'markers':len(chosen),'median_native_rc':float(np.nanmedian(v.rc)) if np.isfinite(v.rc).any() else None,'rc_p25':float(np.nanquantile(v.rc,.25)) if np.isfinite(v.rc).any() else None,'rc_p75':float(np.nanquantile(v.rc,.75)) if np.isfinite(v.rc).any() else None,'median_native_bias':float(np.nanmedian(v.bias)) if np.isfinite(v.bias).any() else None})
        rep=[n for n in NAMES if str(clusters.loc[n,'is_representative'])=='True'];index=[NAMES.index(n) for n in rep]
        s['deduplicated']={'representatives':len(rep),'finite':int(np.isfinite(a[index]).sum()),'median_icc':float(np.nanmedian(a[index])),'median_ci':ci(np.nanmedian(bb[:,index],axis=1))}
        summary.append(s)
    excludedrows=[{'population':g,'marker_id':n,'technical_status':'technisch_nicht_reproduzierbar_Entwicklungsaequivalenz','marker_anderweitig_testgemessen':n in PRELOADED,'F22_historical_provenance_status':'offen'} for g in masks for n in sorted(EXCLUDED)]
    pd.concat([frame,pd.DataFrame(excludedrows)],ignore_index=True).to_csv(OUT/'repeat_marker.csv',index=False)
    checkpoint('W_summary.json',summary);pd.DataFrame(family).to_csv(OUT/'W_family_unit_secondary.csv',index=False)
    checkpoint('W_analysis_complete.json',{'freeze':sha(OUT/'statistical_freeze.json'),'outputs':{n:sha(OUT/n) for n in ['W_summary.json','repeat_marker.csv','W_family_unit_secondary.csv']}})
def m_analysis():
    source=OUT/'M_features_private.parquet';selection=read(OLD/'m_selection_private.parquet',['subject_id','study_id','half','label','qrs_stratum','rhythm_stratum']).sort_values('subject_id')
    assert len(selection)==5206 and not selection.subject_id.duplicated().any()
    ids=[selection.loc[selection.half.eq(h),'subject_id'].to_numpy() for h in (0,1)]
    checkpoint('M_bootstrap_population_private.json',[v.tolist() for v in ids])
    matrix=np.full((1000,len(NAMES)),3,np.int8);halfboot=np.full((2,1000,len(NAMES)),3,np.int8);points=np.full(len(NAMES),3,np.int8);halfpoints=np.full((2,len(NAMES)),3,np.int8);rows=[]
    for block in range(0,len(NAMES),16):
        e.a.guard();names=NAMES[block:block+16]
        dest=OUT/'M_statistics_checkpoints';dest.mkdir(exist_ok=True);p=dest/f'block_{block:04d}.npz';rowpath=dest/f'block_{block:04d}.csv';bindpath=dest/f'block_{block:04d}_binding.json'
        if bindpath.exists():
            bound=e.readsmall(bindpath);assert bound['freeze']==sha(OUT/'statistical_freeze.json') and bound['names']==names
            assert sha(p)==bound['output_sha256'] and sha(rowpath)==bound['row_sha256']
            saved=np.load(p);bout=saved['half_bootstrap_classes'];halfpoints[:,block:block+len(names)]=saved['half_points'];saved.close()
            halfboot[:,:,block:block+len(names)]=bout;matrix[:,block:block+len(names)]=combine(bout[0],bout[1]);points[block:block+len(names)]=combine(halfpoints[0,block:block+len(names)],halfpoints[1,block:block+len(names)])
            rows.extend(pd.read_csv(rowpath,float_precision='round_trip').to_dict('records'));continue
        blockrows=[]
        f=read(source,['subject_id','study_id',*[n+s for n in names for s in ('_kors','_dower')]]).set_index('study_id').reindex(selection.study_id)
        bout=np.full((2,1000,len(names)),3,np.int8)
        for j,n in enumerate(names):
            for h in (0,1):
                mask=selection.half.eq(h).to_numpy();x=f[n+'_kors'].to_numpy(float)[mask];y=f[n+'_dower'].to_numpy(float)[mask]
                code,met=weighted_metrics(x,y,np.ones((1,len(x))))
                halfpoints[h,block+j]=code[0];row={'marker_id':n,'half':h,'class':CODE_NAMES[code[0]],'n_total':len(x),'n_kors':int(np.isfinite(x).sum()),'n_dower':int(np.isfinite(y).sum()),**{k:float(v[0]) for k,v in met.items()}}
                row['loa_low']=row['bias']-1.96*row['difference_sd'];row['loa_high']=row['bias']+1.96*row['difference_sd']
                valid=np.isfinite(x)&np.isfinite(y);delta=y[valid]-x[valid]
                row['distinct_kors']=len(np.unique(x[valid]));row['distinct_dower']=len(np.unique(y[valid]))
                row['quantile_low']=float(np.quantile(delta,.025)) if len(delta) else np.nan;row['quantile_high']=float(np.quantile(delta,.975)) if len(delta) else np.nan
                bootmetrics={k:np.full(1000,np.nan) for k in met}
                for lo,w in weights(ids[h],1461202+h):
                    cls,bm=weighted_metrics(x,y,w);bout[h,lo:lo+20,j]=cls
                    for k,v in bm.items():bootmetrics[k][lo:lo+20]=v
                for k,v in bootmetrics.items():
                    c=ci(v);row.update({k+'_ci_low':c['low'],k+'_ci_high':c['high'],k+'_bootstrap_valid':c['valid']})
                blockrows.append(row)
        halfboot[:,:,block:block+len(names)]=bout
        matrix[:,block:block+len(names)]=combine(bout[0],bout[1]);points[block:block+len(names)]=combine(halfpoints[0,block:block+len(names)],halfpoints[1,block:block+len(names)])
        rows.extend(blockrows)
        assert not p.exists() and not rowpath.exists();np.savez_compressed(p,names=np.array(names),half_bootstrap_classes=bout,half_points=halfpoints[:,block:block+len(names)]);pd.DataFrame(blockrows).to_csv(rowpath,index=False)
        checkpoint(f'M_statistics_checkpoints/block_{block:04d}_binding.json',{'freeze':sha(OUT/'statistical_freeze.json'),'names':names,'seeds':[1461202,1461203],'output_sha256':sha(p),'row_sha256':sha(rowpath)})
        e.a.save('progress_statistics.json',{'phase':'M','markers_completed':block+len(names),'total':len(NAMES)});gc.collect()
    excludedrows=[{'marker_id':n,'half':h,'class':'technisch_nicht_reproduzierbar_Entwicklungsaequivalenz','n_total':len(ids[h])} for n in sorted(EXCLUDED) for h in (0,1)]
    pd.DataFrame(rows+excludedrows).to_csv(OUT/'matrix_half_marker.csv',index=False)
    prerows=pd.DataFrame(rows);prerows.loc[prerows.marker_id.isin(PRELOADED)].to_csv(OUT/'matrix_preloaded_markers.csv',index=False)
    dev=pd.read_csv(ROOT/'nachrechnung/transform_katalog.csv').set_index('marker_id');codes={n:i for i,n in enumerate(CODE_NAMES)}
    idx=[NAMES.index(n) for n in U['D_Mclass']];blind=[NAMES.index(n) for n in NAMES if n not in PRELOADED];dependent=points[blind]==2
    devcodes=np.array([codes[dev.loc[n,'class']] for n in U['D_Mclass']]);point_agree,point_kappa=agreement(devcodes,points[idx])
    boot=[agreement(devcodes,r[idx]) for r in matrix];agr=ci([r[0] for r in boot]);kap=ci([r[1] for r in boot]);drift=ci((matrix[:,blind]==2).sum(1)/2000-.534)
    summary={'coverage_denominator':len(idx),'coverage_numerator':int((points[idx]<3).sum()),'coverage':float((points[idx]<3).mean()),'agreement':point_agree,'agreement_ci':agr,'kappa':point_kappa,'kappa_ci':kap,'primary_universe':2000,'transform_dependent':int(dependent.sum()),'proportion':float(dependent.mean()),'development_proportion':.534,'proportion_drift':float(dependent.mean()-.534),'drift_ci':drift,'full_catalog_universe_after_technical_exclusion':2003,'full_catalog_proportion':float((points==2).sum()/2003),'original_denominator_sensitivity':{'universe':2002,'dependent':int(dependent.sum()),'proportion':float(dependent.sum()/2002),'development_proportion':1069/2002,'drift_ci':ci((matrix[:,blind]==2).sum(1)/2002-1069/2002),'agreement_denominator':1752,'agreement':float(np.sum(points[idx]==devcodes)/1752),'coverage':float(np.sum(points[idx]<3)/1752)},'estimable_bootstrap_min':int((matrix[:,idx]<3).sum(1).min()),'estimable_bootstrap_max':int((matrix[:,idx]<3).sum(1).max()),'estimable_bootstrap_median':float(np.median((matrix[:,idx]<3).sum(1))),'classes':{n:int(np.sum(points==k)) for k,n in enumerate(CODE_NAMES)}}
    comparisons=[]
    for fld in ('class','train_class','validation_class'):
        d=np.array([codes[dev.loc[n,fld]] for n in U['D_Mclass']]);aa,kk=agreement(d,points[idx]);comparisons.append({'development':fld,'agreement':aa,'kappa':kk,'n':len(idx)})
    summary['development_fold_comparisons']=comparisons
    oldpoint=agreement(np.r_[devcodes,2],np.r_[points[idx],3]);oldboot=[agreement(np.r_[devcodes,2],np.r_[r[idx],3]) for r in matrix]
    summary['original_denominator_sensitivity'].update(kappa=oldpoint[1],agreement_ci=ci([v[0] for v in oldboot]),kappa_ci=ci([v[1] for v in oldboot]))
    depidx=[j for j,n in enumerate(NAMES) if n not in set(U['M_independent_variable']+U['M_independent_constant'])|PRELOADED]
    avail=int((points[depidx]<3).sum());summary['availability_conditional_dependent_proportion']={'transform_dependent':int((points[depidx]==2).sum()),'estimable_matrix_dependent':avail,'proportion':float((points[depidx]==2).sum()/avail) if avail else None}
    summary['primary_classes']={n:int(np.sum(points[blind]==k)) for k,n in enumerate(CODE_NAMES)}
    summary['primary_class_fraction_CIs']={n:ci((matrix[:,blind]==k).sum(1)/len(blind)) for k,n in enumerate(CODE_NAMES)}
    summary['full_class_fraction_CIs']={n:ci((matrix==k).sum(1)/len(NAMES)) for k,n in enumerate(CODE_NAMES)}
    ha,hk=agreement(halfpoints[0],halfpoints[1]);hboot=[agreement(halfboot[0,i],halfboot[1,i]) for i in range(1000)]
    summary['half_concordance']={'agreement':ha,'kappa':hk,'n':len(NAMES),'agreement_ci':ci([v[0] for v in hboot]),'kappa_ci':ci([v[1] for v in hboot])}
    confusion=[]
    for panel,d,t in [('development_test',devcodes,points[idx]),('half0_half1',halfpoints[0],halfpoints[1])]:
        for i,ni in enumerate(CODE_NAMES):
            for j,nj in enumerate(CODE_NAMES):
                bootcounts=np.sum((devcodes==i)[None,:]&(matrix[:,idx]==j),axis=1) if panel=='development_test' else np.sum((halfboot[0]==i)&(halfboot[1]==j),axis=1)
                c=ci(bootcounts)
                confusion.append({'panel':panel,'first_class':ni,'second_class':nj,'markers':int(np.sum((d==i)&(t==j))),'ci_low':c['low'],'ci_high':c['high'],'bootstrap_valid':c['valid']})
    pd.DataFrame(confusion).to_csv(OUT/'matrix_confusion.csv',index=False)
    frame=pd.DataFrame({'marker_id':NAMES,'class':[CODE_NAMES[k] for k in points],'half0_class':[CODE_NAMES[k] for k in halfpoints[0]],'half1_class':[CODE_NAMES[k] for k in halfpoints[1]],'development_class':dev.reindex(NAMES)['class'].to_numpy(),'primary_agreement_member':[n in U['D_Mclass'] for n in NAMES],'preloaded':[n in PRELOADED for n in NAMES]})
    frame['same_total_class']=frame['class'].eq(frame.development_class)
    frame['primary_agreement_status']=np.where(frame.primary_agreement_member,np.where(frame['class'].eq('nicht_schaetzbar'),'nicht_schaetzbar',np.where(frame.same_total_class,'Kriterium_erfuellt','nicht_erfuellt')),'ausserhalb_primarem_Klassenuniversum')
    excludedrows=[{'marker_id':n,'class':'technisch_nicht_reproduzierbar_Entwicklungsaequivalenz','half0_class':'technisch_nicht_reproduzierbar_Entwicklungsaequivalenz','half1_class':'technisch_nicht_reproduzierbar_Entwicklungsaequivalenz','development_class':dev.loc[n,'class'],'primary_agreement_member':False,'preloaded':False} for n in sorted(EXCLUDED)]
    pd.concat([frame,pd.DataFrame(excludedrows)],ignore_index=True).to_csv(OUT/'matrix_marker.csv',index=False);checkpoint('M_summary.json',summary)
    # Fixed coefficient sensitivity, paired outputs and all missing statuses.
    values=[];statuscounts={}
    sensitivity_hashes={x['path']:x['sha256'] for x in e.readsmall(OUT/'M_input_binding.json')['sensitivity_checkpoints']}
    for study in selection.study_id:
        p=OUT/'M_sensitivity_private'/f'{int(study)}.json'
        if not p.exists():statuscounts['extraction_or_matrix_abstention']=statuscounts.get('extraction_or_matrix_abstention',0)+1;continue
        assert sha(p)==sensitivity_hashes[str(p)]
        r=e.readsmall(p);assert r['freeze_sha256']==sha(OUT/'extraction_freeze.json')
        statuscounts[r['status']]=statuscounts.get(r['status'],0)+1
        if r['status']!='ok':continue
        a=r['methods']['inverse_dower'];b=r['methods']['plus']
        start=len(values)
        for key in set(a)|set(b):
            av=a.get(key);bv=b.get(key)
            if isinstance(av,dict) or isinstance(bv,dict):
                both=av is not None and bv is not None
                values.append({'output':key,'kind':'turn','both_available':both,'changed':both and av['class']!=bv['class'],'majority_changed':both and av['majority']!=bv['majority'],'cw_ccw_flip':both and {av['majority'],bv['majority']}=={'cw','ccw'}})
            elif isinstance(av,list) or isinstance(bv,list):
                for j in range(3):values.append({'output':key+'_'+str(j),'kind':'numeric','difference':float(bv[j]-av[j]) if av is not None and bv is not None else np.nan})
            else:values.append({'output':key,'kind':'numeric','difference':float(bv-av) if av is not None and bv is not None else np.nan})
        for v in values[start:]:v['study_id']=int(study)
    sec=[]
    if values:
        v=pd.DataFrame(values)
        for key,a in v.groupby('output'):
            r={'output':key,'candidate_ECGs':len(selection),'output_records':len(a)}
            if a.kind.iloc[0]=='turn':r.update(finite_pairs=int(a.both_available.sum()),class_changed=int(a.changed.sum()),majority_changed=int(a.majority_changed.sum()),cw_ccw_flip=int(a.cw_ccw_flip.sum()))
            else:
                d=a.difference.to_numpy(float);d=d[np.isfinite(d)];r.update(finite_pairs=len(d),median_difference=float(np.median(d)) if len(d) else None,p025=float(np.quantile(d,.025)) if len(d) else None,p975=float(np.quantile(d,.975)) if len(d) else None)
            sec.append(r)
    if values:e.a.writep('M_coefficient_raw_differences_private.parquet',pd.DataFrame(values))
    checkpoint('M_sensitivity_statuses.json',statuscounts);pd.DataFrame(sec,columns=['output','candidate_ECGs','output_records','finite_pairs','class_changed','majority_changed','cw_ccw_flip','median_difference','p025','p975']).to_csv(OUT/'M_coefficient_sensitivity.csv',index=False)
    checkpoint('M_analysis_complete.json',{'freeze':sha(OUT/'statistical_freeze.json'),'outputs':{n:sha(OUT/n) for n in ['M_summary.json','matrix_marker.csv','matrix_half_marker.csv','matrix_preloaded_markers.csv','matrix_confusion.csv','M_coefficient_sensitivity.csv','M_sensitivity_statuses.json']}})
def p_analysis():
    f=read(OLD/'p_test_cache_left_join_private.parquet',['subject_id','study_id','split','p_duration_ms','extraction_status']).sort_values('subject_id')
    assert len(f)==1500 and not f.subject_id.duplicated().any() and f.split.eq('locked_test').all()
    x=f.p_duration_ms.to_numpy(float);finite=np.isfinite(x);n=int(finite.sum());vals=x[finite]
    point=np.quantile(vals,[.25,.5,.75,.95,.975],method='linear') if n else np.full(5,np.nan)
    bb=np.full((1000,6),np.nan)
    # Literal patient draws preserve numpy's linear order-statistic interpolation.
    rng=np.random.Generator(np.random.PCG64(1461204))
    for i in range(1000):
        a=x[rng.integers(0,len(x),len(x))];v=a[np.isfinite(a)];bb[i,0]=len(v)/len(x)
        if len(v):bb[i,1:]=np.quantile(v,[.25,.5,.75,.95,.975],method='linear')
    counts=np.rint(bb[:,0]*1500)
    result={'selected':len(x),'available':n,'missing':len(x)-n,'availability':n/len(x),'availability_ci':ci(bb[:,0]),'development_availability':932/1500,'availability_drift':n/len(x)-932/1500,'availability_drift_ci':ci(bb[:,0]-932/1500),'finite_nonpositive_durations':int(np.sum(vals<=0)),'strata':{'lt120':int(np.sum(vals<120)),'120_to_160_inclusive':int(np.sum((vals>=120)&(vals<=160))),'gt160':int(np.sum(vals>160)),'ge120':int(np.sum(vals>=120))},'source_kind':'AW159 release re-extraction cache, not a pre-existing historical test cache','finite_bootstrap_min':int(np.min(counts)),'finite_bootstrap_max':int(np.max(counts)),'finite_bootstrap_median':float(np.median(counts))}
    for j,k in enumerate(('p25','median','p75','p95','p975')):result[k]=float(point[j]);result[k+'_ci']=ci(bb[:,j+1])
    checkpoint('P_summary.json',result);np.savez_compressed(OUT/'P_bootstrap_checkpoint.npz',values=bb,seed=1461204,freeze=sha(OUT/'statistical_freeze.json'))
    checkpoint('P_analysis_complete.json',{'freeze':sha(OUT/'statistical_freeze.json'),'output_sha256':sha(OUT/'P_summary.json'),'bootstrap_sha256':sha(OUT/'P_bootstrap_checkpoint.npz')})
def run():
    resume=e.PHASE=='statistics_resume'
    assert resume or not (OUT/'statistics_started.json').exists(),'One-time gate: no second computation'
    assert not (OUT/'statistics_complete.json').exists(),'Do not restart a completed computation'
    freeze=e.readsmall(OUT/'statistical_freeze.json')
    assert all(sha(p)==h for p,h in freeze['code'].items())
    assert all(sha(p)==h for p,h in freeze['inputs'].items())
    if resume:
        assert e.readsmall(OUT/'statistics_started.json')['freeze']==sha(OUT/'statistical_freeze.json')
        e.public('Technical statistics resumption',f'Statistics freeze `{sha(OUT/"statistical_freeze.json")}`; only valid own checkpoints, no second fit')
    else:
        checkpoint('statistics_started.json',{'utc':e.a.utc(),'freeze':sha(OUT/'statistical_freeze.json')})
        e.public('Start of one-time replication computation',f'Statistics freeze `{sha(OUT/"statistical_freeze.json")}`; W then M then P, unchanged gates')
    for kind,func in [('W',w_analysis),('M',m_analysis),('P',p_analysis)]:
        p=OUT/f'{kind}_analysis_complete.json'
        if p.exists():
            bound=e.readsmall(p);assert bound['freeze']==sha(OUT/'statistical_freeze.json')
            if kind=='P':assert bound['output_sha256']==sha(OUT/'P_summary.json') and bound['bootstrap_sha256']==sha(OUT/'P_bootstrap_checkpoint.npz')
            else:assert all(sha(OUT/n)==h for n,h in bound['outputs'].items())
        else:func()
    checkpoint('statistics_complete.json',{'utc':e.a.utc(),'freeze':sha(OUT/'statistical_freeze.json'),'result_view_authorized':True})
def main():
    e.PHASE=sys.argv[1]
    e.a.save(f'status_{e.PHASE}.json',{'state':'running','pid':os.getpid()})
    try:
        with e.Budget():
            e.a.guard()
            if e.PHASE=='freeze':freeze()
            else:run()
        e.a.save(f'status_{e.PHASE}.json',{'state':'complete','utc':e.a.utc()})
    except Exception:
        e.a.save(f'status_{e.PHASE}.json',{'state':'STOP','traceback_private':traceback.format_exc()});raise
if __name__=='__main__':main()
