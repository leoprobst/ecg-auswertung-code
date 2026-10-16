"""AW105: vollständige train-Auswahlroutine; fehlender Regelvertrag stoppt Patientenfit."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .models import Fit,draw
from werkzeuge.nachrechnung.core import auc,interval
from .runner import small,save,sha,PROJECT
from .clinical import reference,eligible

def lsb_masks(qrs,q_i,q_v6,s_v1,r_v1,s_v6,time_v5,time_v6):
    """Explizite AHA-Release-/WHO-R-Zeitvariante; Unknown in beiden gleich.

    Entspricht den im Release benutzten Höhen/Tiefen; keine vollständige
    LSB-Adjudikation. Die erforderlichen q-Dauern/R-Zeiten fehlen im Cache.
    """
    arrays=[np.asarray(v,float) for v in (qrs,q_i,q_v6,s_v1,r_v1,s_v6,time_v5,time_v6)]
    known=np.isfinite(np.column_stack(arrays)).all(axis=1)
    common=(arrays[0]>=120)&(arrays[1]<15)&(arrays[2]<15)&(arrays[3]>arrays[4])&(arrays[5]<.10)
    primary=common&(arrays[6]>60)&(arrays[7]>60)
    who=common&((arrays[6]>=60)|(arrays[7]>=60))
    return np.where(known,primary,np.nan),np.where(known,who,np.nan)

def basis(f):
    a=f.age.to_numpy(float);sex=f.sex_numeric.to_numpy(float);q=f.qrs_duration_ms.to_numpy(float)
    return np.column_stack([a,np.maximum(a-40,0),np.maximum(a-60,0),np.maximum(a-80,0),sex,f.heart_rate,q,f.qtc_fridericia_ms,q*sex])

def candidates(f):
    r=lambda l:f['lead_measurement_'+l+'_qrs_r_height_mv'].to_numpy(float)
    s=lambda l:f['lead_measurement_'+l+'_qrs_s_depth_mv'].to_numpy(float)
    return {'C01':f.classical_cornell_product.to_numpy(float),'C02':f.classical_cornell_voltage.to_numpy(float),
      'C03':f.classical_sokolow_lyon.to_numpy(float),'C04':s('V2')+r('V6'),'C05':r('aVL'),'C06':r('I')+s('III'),
      'C07':r('I')+s('III')-r('III')-s('I'),'C08':None}

def inner_folds(y):
    out=np.zeros(len(y),int)
    for k,(_,v) in enumerate(StratifiedKFold(5,shuffle=True,random_state=105).split(np.zeros(len(y)),y)):out[v]=k
    return out

def select_fit(x,cand,peak,y,validation_x,validation_cand,validation_peak,folds,w=None):
    w=np.ones(len(y)) if w is None else w;scores={}
    for name,value in cand.items():
        z=x if value is None else np.column_stack([x,value]);pred=np.zeros(len(y))
        for k in range(5):
            tr=folds!=k;va=~tr;fit=Fit().fit(z[tr],y[tr],w[tr]);pred[va]=fit.score(z[va])
        scores[name]=auc(y,pred,w)
    maximum=max(scores.values());winner=next(k for k in sorted(scores) if scores[k]>=maximum-1e-6)
    tr=x if cand[winner] is None else np.column_stack([x,cand[winner]])
    va=validation_x if validation_cand[winner] is None else np.column_stack([validation_x,validation_cand[winner]])
    base=Fit().fit(tr,y,w);ext=Fit().fit(np.column_stack([tr,peak]),y,w)
    return winner,scores,base.score(va),ext.score(np.column_stack([va,validation_peak])),ext

def estimate(f,y,cand,replicates=1000):
    from .models import EXTRA
    assert not f.subject_id.duplicated().any()
    tr=f.split.eq('train').to_numpy();va=~tr;x=basis(f);peak=f.spatial_low_voltage_qrs_peak_magnitude.to_numpy(float);y=np.asarray(y,float)
    ct={k:None if v is None else v[tr] for k,v in cand.items()};cv={k:None if v is None else v[va] for k,v in cand.items()}
    folds=inner_folds(y[tr]);winner,score,s0,s1,ext=select_fit(x[tr],ct,peak[tr],y[tr],x[va],cv,peak[va],folds)
    point=[auc(y[va],s0),auc(y[va],s1)];point.append(point[1]-point[0]);rng=np.random.default_rng(1052026);boot=[];freq={k:0 for k in cand};failures={}
    for _ in range(replicates):
        wt=draw(y[tr],rng);wv=draw(y[va],rng)
        try:
            selected,_,s0,s1,_=select_fit(x[tr],ct,peak[tr],y[tr],x[va],cv,peak[va],folds,wt)
            b=auc(y[va],s0,wv);a=auc(y[va],s1,wv);boot.append([b,a,a-b]);freq[selected]+=1
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:failures[str(e)]=failures.get(str(e),0)+1
    result={'status':'completed' if len(boot)>=.95*replicates else 'bootstrap_unzureichend','selected':winner,'inner_oof_auc':score,'baseline_auc':point[0],'augmented_auc':point[1],'delta_auc':point[2],
      'train_marker_beta':ext.raw_marker_beta(),'selection_frequencies':freq,'successful':len(boot),'requested':replicates,'failures':failures,'inner_duplicate_rule':'Originalpersonen bleiben in demselben inneren Fold; Multiplizitätsgewichte statt duplizierter Zeilen'}
    if boot:result.update({name+'_ci95':interval(np.asarray(boot)[:,j]) for j,name in enumerate(('baseline_auc','augmented_auc','delta_auc'))})
    return result

def run(out):
    f=small(out/'lvh_index_private.parquet');e=small(out/'echo_evidence_private.parquet');f=reference(f,e,'echo_lvh_strict')
    from werkzeuge.nachrechnung.core import assert_development
    assert_development(f);assert not f.subject_id.duplicated().any()
    pre=small(out/'preindex.parquet',columns=['subject_id','split','classical_lbbb_pattern_positive'])
    assert set(f.subject_id)<=set(pre.subject_id)
    ref=eligible(f,'echo_lvh_strict');lsb=f.classical_lbbb_pattern_positive.eq(True);complete=np.isfinite(f[['age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms','spatial_low_voltage_qrs_peak_magnitude']]).all(axis=1)
    cand=candidates(f);joint=complete.copy()
    for v in cand.values():
        if v is not None:joint &= np.isfinite(v)
    flows={}
    for split in ('train','validation'):
        m=f.split.eq(split);z=m&lsb&ref&joint;y=f['echo_lvh_strict__label_state'].eq('positive')
        flows[split]={'all_index':int(m.sum()),'LSB_positive':int((m&lsb).sum()),'LSB_unknown':int((m&f.classical_lbbb_pattern_positive.isna()).sum()),'LSB_echo':int((m&lsb&f.nearest_tte_delta_hours.abs().le(720)).sum()),'LSB_mass_eligible':int((m&lsb&ref).sum()),'joint_C01_C08_n':int(z.sum()),'positive':int((z&y).sum()),'negative':int((z&~y).sum()),'LSB_mass_7d':int((m&lsb&eligible(f,'echo_lvh_strict',168)).sum()),'paced_report_positive':int((m&lsb&f.current_report_pacemaker_state.eq('positive')).sum())}
        initial=pre.split.eq(split);flows[split].update({'first_technical_n':int(initial.sum()),'first_technical_block_unknown':int((initial&pre.classical_lbbb_pattern_positive.isna()).sum()),'excluded_no_interpretable_block':int(initial.sum()-m.sum())})
    gates=['C09 Kafka und C10 native atriale Regel nicht vollständig verifiziert/implementiert; keine Recherche in AW121','Sicherer signalbasierter Pacing-/Präexzitationsausschluss fehlt','WHO-Variante nicht aus gespeicherten q-Dauern/V5-/V6-R-Zeiten rekonstruierbar']
    if any(min(v['positive'],v['negative'])<100 for v in flows.values()):gates.append('Weniger als 100 Personen je Klasse/Fold auf verfügbarer C01-C08-Basis')
    result={'status':'nicht_pruefbar_vertragstor','delta_auc':None,'reason':gates,'counts':flows,'available_candidates':list(cand),'candidate_missingness':{k:int((lsb&ref&~np.isfinite(v)).sum()) if v is not None else 0 for k,v in cand.items()},'primary_fit_executed':False,'claim':'Machbarkeit, kein biologischer Nullbefund'}
    # Optional later K-attested native rules, same run-id and unchanged code.
    # No invented C09/C10 formulas or report-derived pacing proxy.
    contract=out/'AW105_rules_attestation.json';inputs=out/'AW105_attested_inputs.parquet'
    if contract.exists() and inputs.exists():
        import json
        att=json.loads(contract.read_text(encoding='utf8'))
        assert att['status']=='verified_by_K' and att['scope']=='train_validation_only'
        assert att['spec_sha256']==sha(PROJECT/'nachrechnung/SPEZIFIKATION_AW105_LVH_LSB.md')
        assert att['input_sha256']==sha(inputs)
        assert all(att['rules_verified'][k] for k in ('C09_original_fulltext','C10_native_atrial','signal_pacing_exclusion','preexcitation_exclusion','WHO_variant'))
        z=small(inputs,columns=['subject_id','study_id','split','C09','C10','pacing_free_signal','preexcitation_free_signal','WHO_LSB_variant'])
        assert set(z.split)<={'train','validation'} and not z.study_id.duplicated().any()
        f=f.merge(z,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
        cand=candidates(f);cand.update({k:f[k].to_numpy(float) for k in ('C09','C10')})
        allowed=lsb&ref&joint&f.pacing_free_signal.eq(True)&f.preexcitation_free_signal.eq(True)&np.isfinite(f.C09)&np.isfinite(f.C10)
        picked=f[allowed].copy();y=picked['echo_lvh_strict__label_state'].eq('positive').to_numpy(float)
        count={s:{'n':int(m.sum()),'positive':int(y[m].sum()),'negative':int((y[m]==0).sum())} for s in ('train','validation') for m in [picked.split.eq(s).to_numpy()]}
        if all(min(v['positive'],v['negative'])>=100 for v in count.values()):
            result=estimate(picked,y,{k:None if v is None else v[allowed] for k,v in cand.items()})
            result.update({'counts':flows,'counts_primary':count,'reason':[],'primary_fit_executed':True,'contract_sha256':sha(contract),'input_sha256':sha(inputs)})
        else:result.update({'reason':['vollständiger attestierter Vertrag vorhanden, Mindestfallzahl nicht erreicht'],'counts_primary':count})
    save(out/'lvh_result.json',result);print(__import__('json').dumps({'phase':'LVH_gate','status':result['status'],'counts':flows,'gates':result['reason']}),flush=True)

def synthetic():
    rng=np.random.default_rng(105);x=rng.normal(size=(300,2));y=(x[:,0]+rng.normal(size=300)>0).astype(float);p=rng.normal(size=300)
    c={'C01':x[:,0]+rng.normal(size=300)*.2,'C02':rng.normal(size=300),'C08':None};fold=inner_folds(y)
    a=select_fit(x,c,p,y,x,c,p,fold)
    b=select_fit(x,c,p,y,-x,c,-p,fold)
    assert a[0]==b[0] and a[1]==b[1]
    wt=draw(y,rng);again=select_fit(x,c,p,y,x,c,p,fold,wt)
    assert np.isfinite(again[2]).all()
    a,b=lsb_masks([120]*4,[0]*4,[0]*4,[1]*4,[.5]*4,[0]*4,[60,60.000001,59.999999,np.nan],[61]*4)
    assert np.array_equal(a[:3],[0,1,0]) and np.array_equal(b[:3],[1,1,1]) and np.isnan(a[3]) and np.isnan(b[3])
    return {'status':'passed','selection_train_only':True,'bootstrap_multiplizities_single_inner_fold':True,'candidate_tie_order':'C01..C10 within 1e-6','WHO_AHA_60ms_unknown_tests':True}
