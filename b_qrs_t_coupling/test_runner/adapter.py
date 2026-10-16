"""v4-Analyseadapter; ausschließlich auf explizit übergebener Indexpopulation.

Die Funktionen lesen keine Dateien. Ein fehlendes Teilattest erzeugt einen
sichtbaren STOP; es wird weder ein Ersatzmarker noch ein Ersatzindex gewählt.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from .daten import masks,classify_lvef,HYPOTHESES,truth,select_index
from .statistik import raw_design,fixed_design,fit,holm,separation
from werkzeuge.nachrechnung.core import logistic_fit
from werkzeuge.nachrechnung.resources import guard

FALLEN={'F1':'eingeschraenkt','F2':'eingeschraenkt','F3':'nicht_zutreffend','F4':'eingeschraenkt','F5':'offen','F6':'eingeschraenkt','F7':'offen','F8':'offen'}

def stopped(reason):
    return {'status':'nicht_schaetzbar','urteil':'nicht_schaetzbar','p_raw':None,'reason':str(reason),'fallen':FALLEN}

def estimate(f,entry,union,selector=None,inclusive=False,strong=False,marker=None,parameter=None,bootstrap=True,weights=None,sex_stratum=None):
    from .runner import bootstrap as resample
    marker=marker or entry['marker'];ep=entry['endpoint'];comp=entry['comparator']
    try:
        if entry.get('_require_measurement') and entry['measurement_attestation'].get('status')!='bestanden':raise ValueError('endpunktspezifisches Messtor nicht attestiert')
        z,good,flow=masks(f,ep,marker,comp,strong=strong)
        if selector is not None:good &= selector(z).fillna(False)
        x,names=raw_design(z.loc[good],marker,comp,strong=strong)
        params=parameter or entry['scaling_B4' if strong else 'scaling_B3']
        if params.get('status')!='gebunden':raise ValueError('Skalierung fehlt: '+params.get('reason','ungebunden'))
        if names!=params['names']:raise ValueError('Designspalten/Parameter nicht identisch')
        if sex_stratum is not None:
            if sex_stratum not in ('F','M') or not z.loc[good,'sex'].eq(sex_stratum).all():raise ValueError('Geschlechtsstratum nicht homogen F/M')
            # §0 Nachtrag 2(a): strukturell, vor Fit/Bootstrap; keine neue SD.
            idx=[i for i,n in enumerate(names) if n!='sex_numeric']
            x=x[:,idx]
            params=dict(params,**{k:[params[k][i] for i in idx] for k in ('names','mean','sd','keep')})
        design=fixed_design(x,params)
        y=classify_lvef(z.loc[good,'preferred_lvef'],inclusive) if ep=='reduced_lvef_strict' else z.loc[good,'label_state'].eq('positive').to_numpy(float)
        ids=z.loc[good,'subject_id'].to_numpy()
        w=None if weights is None else np.asarray(weights)[good.to_numpy()]
        result=fit(design,y,ids,entry['direction'],weights=w)
        result.update(status='schaetzbar',fallen=FALLEN,missingness_fallfluss=flow,n_selected=int(good.sum()))
        if sex_stratum is not None:result.update(sex_stratum=sex_stratum,structural_omission='sex_numeric',design_names=params['names'])
        result['bootstrap']=resample(design,y,ids,union,weights=w) if bootstrap else {'status':'nicht_ausgefuehrt','reason':'nur explizite synthetische Einzelpfadprüfung'}
        return result
    except (ValueError,KeyError,AttributeError,RuntimeError,np.linalg.LinAlgError) as exc:
        result=stopped(exc)
        if 'flow' in locals():result.update(missingness_fallfluss=flow,n_selected=int(good.sum()))
        return result

def nt_numeric(f):
    fields=['ntprobnp_nearest','ntprobnp_reference_upper','ntprobnp_delta_hours','ntprobnp_nearest_censoring','ntprobnp_unit','ehr_time_alignment_confirmatory_eligible']
    if set(fields)-set(f):raise ValueError('NT-Einheiten-/Zensierungsvertrag fehlt')
    valid=(f.ntprobnp_nearest.ge(0)&np.isfinite(f.ntprobnp_nearest)&f.ntprobnp_reference_upper.gt(0)&np.isfinite(f.ntprobnp_reference_upper)
           &f.ntprobnp_delta_hours.between(-6,12)&f.ntprobnp_nearest_censoring.eq('exact')&f.ntprobnp_unit.isin(['pg/mL','ng/L'])&truth(f.ehr_time_alignment_confirmatory_eligible))
    return np.log1p(f.ntprobnp_nearest/f.ntprobnp_reference_upper).where(valid)

def lvh_selector(f):
    from werkzeuge.b_vorbereitung.clinical import eligible
    lv=eligible(f,'echo_lvh_strict')&f['echo_lvh_strict__label_state'].eq('positive')
    return f.preferred_lvef.le(40)|(f.preferred_lvef.ge(50)&lv)

def rv_selector(f):
    from werkzeuge.b_vorbereitung.clinical import eligible
    return eligible(f,'reduced_lvef_strict')&f.lvef_conflict_gt_10.eq(False).fillna(False)&f.preferred_lvef.between(50,100)

def residual(values,numeric,sex,package):
    p=package['parameters'];x=np.asarray(numeric,float);sex=np.asarray(sex)
    if not np.isfinite(x).all() or any(s not in p['categories'] for s in sex):raise ValueError('Residualkovariaten unbekannt')
    design=np.column_stack([(x-np.asarray(p['mean']))/np.asarray(p['sd'])]+[(sex==c).astype(float) for c in p['categories']])
    return np.asarray(values)-design@np.asarray(p['coef'])-p['intercept']

def signal_values(row,expected,residual_package):
    """Releaseadapter für jeden unveränderten Index; keine Pilotneuauswahl."""
    from werkzeuge.varianten.adapter import prepare_record,context,release_import
    from werkzeuge.b_vorbereitung.kernels import coupling,LEADS
    from werkzeuge.varianten.metriken import form_distance
    release_import()
    from vcg_biomarker.core.biomarkers.discovery_v3_biomarkers import _qrs_t_distances
    guard();package=prepare_record(row)
    if package['fs']!=500:raise ValueError('500-Hz-Messvertrag verletzt')
    c=context(package);d=context(package,'inverse_dower');original=_qrs_t_distances(c);dower=_qrs_t_distances(d)
    if original is None or not np.isclose(original['normalized_discrete_frechet'],expected,rtol=1e-8,atol=0):raise ValueError('Originaläquivalenzgate')
    q=c.vcg_segment_origin_corrected('qrs');t=c.vcg_segment_origin_corrected('t')
    q8=np.column_stack([c.ecg_segment(l,'qrs') for l in LEADS]);t8=np.column_stack([c.ecg_segment(l,'t') for l in LEADS])
    result={'original':original['normalized_discrete_frechet'],'dower':np.nan if dower is None else dower['normalized_discrete_frechet'],
            'direct8_centered_rms':coupling(q8,t8),'direct8_qrs_max_centered':coupling(q8,t8,'qrs_max'),
            'kors_qrs_max_centered':coupling(q,t,'qrs_max'),'shape':form_distance(c.vcg_segment('qrs'),c.vcg_segment('t'),'rotation')}
    if not np.isclose(coupling(q,t),result['original'],rtol=1e-8,atol=0):raise ValueError('RMS-Kernäquivalenzgate')
    result['residual']=float(residual([result['original']],[[row[k] for k in ['qrs_duration_ms','qt_interval_ms','context_heart_rate_bpm','age']]],[row['sex']],residual_package)[0])
    return result

def verification_design(f,p):
    x,names=raw_design(f.assign(__verification=0.),'__verification',strong=True);x=x[:,:-1]
    if names[:-1]!=p['names']:raise ValueError('Verifikationsdesign verändert')
    keep=np.asarray(p['keep'],bool);sd=np.asarray(p['sd']);mean=np.asarray(p['mean'])
    if not np.isfinite(x).all() or (sd[keep]<=0).any():raise ValueError('Verifikationsbasis unvollständig')
    if np.any(x[:,~keep]!=mean[~keep]):raise ValueError('Verifikationskonstante verändert')
    return np.column_stack([np.ones(len(x)),(x[:,keep]-mean[keep])/sd[keep]])

def ipw_weights(vx,parameters,beta=None,cap=None):
    prop=expit(vx@np.asarray(parameters['beta'] if beta is None else beta))
    if not np.isfinite(prop).all() or (prop<=0).any():raise ValueError('Propensity nicht positiv/endlich')
    w=np.minimum(1/prop,parameters['p99_train_verified'] if cap is None else cap)
    return prop,w

def ess(weights,y):
    out={str(k):float(np.sum(weights[y==k])**2/np.sum(weights[y==k]**2)) if np.any(y==k) else 0. for k in (0,1)}
    if min(out.values())<100:raise ValueError('ESS<100 je Klasse')
    return out

def verification_associations(technical,entry,params,union,bootstrap=True):
    from .runner import bootstrap as resample
    x,names=raw_design(technical,entry['marker'],strong=True);good=np.isfinite(x).all(axis=1)
    if names!=params['names']:raise ValueError('Verifikationsmarker-Designbindung fehlt')
    z=fixed_design(x[good],params);y=technical.loc[good,'has_tte_30d'].eq(True).to_numpy(float);ids=technical.loc[good,'subject_id'].to_numpy()
    result=fit(z,y,ids)
    if bootstrap:result['bootstrap']=resample(z,y,ids,union)
    # Registrierter Verifikationskomparator, unveränderte MIMIC-Skalierung.
    chosen=[0,1,3,5]  # intercept, age_linear, sex_numeric, measured QRS
    # Name-based indexing accounts for structural development constants.
    kept=[n for n,k in zip(params['names'],params['keep']) if k]
    chosen=[0]+[1+kept.index(n) for n in ['age_linear','sex_numeric','qrs_duration_ms']]
    comp=z[:,chosen];b=logistic_fit(comp,y,np.ones(len(y)))
    from werkzeuge.nachrechnung.core import auc
    return {'status':'schaetzbar','Marker_plus_B4':result,'comparator_age_sex_QRS':{'auc':auc(y,comp@b),'design':['age_linear','sex_numeric','qrs_duration_ms'],'type':'apparente AUC'},
            'availability_30d':int(y.sum()),'availability_7d':int(technical.has_tte_7d.eq(True).sum()),'limitations':'Verifikationsassoziation ersetzt das markerfreie Gewichtsmodell nicht','fallen':FALLEN}

def ipw_replicate(dev_x,dev_y,dev_ids,eval_x,eval_ids,out_ids,r,parameters):
    """Getrennte Personenuniversen; gemeinsame Multiplizität im Outcome-Teil."""
    if np.intersect1d(dev_ids,eval_ids).size:raise ValueError('IPW Entwicklungs-/Prüfpersonen nicht disjunkt')
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1282026,int(r)])))
    def sample(ids):
        ids=np.asarray(ids)
        if not np.array_equal(ids,np.unique(ids)):raise ValueError('IPW Universum nicht sortiert/eindeutig')
        return np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids)).astype(float)
    dw=sample(dev_ids);ew=sample(eval_ids);active=dw>0
    if set(np.unique(dev_y[active]))!={0,1} or np.linalg.matrix_rank(dev_x[active])!=dev_x.shape[1]:raise ValueError('Verifikationsbootstrap Klasse/Rang')
    if separation(dev_x[active],dev_y[active])!='keine':raise ValueError('Verifikationsbootstrap Separation')
    beta=logistic_fit(dev_x[active],dev_y[active],dw[active])
    train_prop=expit(dev_x@beta)
    if not np.isfinite(train_prop).all() or (train_prop<=0).any():raise ValueError('Train-Propensity ungültig')
    verified=(dev_y==1)&active
    take=np.repeat(1/train_prop[verified],dw[verified].astype(int))
    if not len(take):raise ValueError('keine verifizierten train-Personen')
    cap=float(np.quantile(take,.99,method='linear'));prop,weights=ipw_weights(eval_x,parameters,beta,cap)
    positions=np.searchsorted(eval_ids,out_ids)
    if (positions>=len(eval_ids)).any() or not np.array_equal(eval_ids[positions],out_ids):raise ValueError('Outcome außerhalb technischen Universums')
    return ew[positions],weights[positions],{'p99_train':cap,'dev_mult':dw,'eval_mult':ew}

def ipw_analysis(development,technical,entry,parameters,union,bootstrap=True):
    """B3-Outcomefit; B4-Verifikation bleibt unabhängig von Marker-Missingness."""
    try:
        technical=technical.sort_values('subject_id').copy();development=development.sort_values('subject_id').copy()
        if technical.subject_id.duplicated().any() or development.subject_id.duplicated().any():raise ValueError('IPW Mehrfachindex')
        before=len(technical)
        x,_=raw_design(technical.assign(__verification=0.),'__verification',strong=True)
        technical=technical.loc[np.isfinite(x[:,:-1]).all(axis=1)].copy()
        if (technical.has_tte_30d.isna()&(technical.nearest_tte_measurement_id.notna()|technical.nearest_tte_delta_hours.notna())).any():raise ValueError('unbelegte sparse Echo-Nulls')
        vx=verification_design(technical,parameters);dx=verification_design(development,parameters)
        dy=development.has_tte_30d.eq(True).to_numpy(float)
        z,good,flow=masks(technical.assign(index_selection_strategy='first_verified_ecg_reference_result_blind'),entry['endpoint'],entry['marker'],entry['comparator'])
        x,names=raw_design(z.loc[good],entry['marker'],entry['comparator']);design=fixed_design(x,entry['scaling_B3'])
        y=classify_lvef(z.loc[good,'preferred_lvef']);ids=z.loc[good,'subject_id'].to_numpy()
        prop,w=ipw_weights(vx,parameters);weights=w[good.to_numpy()];effective=ess(weights,y)
        unweighted=fit(design,y,ids);weighted=fit(design,y,ids,weights=weights)
        vals={k:[] for k in ('beta','or','delta_auc','auc_G','auc_G_marker','auc_marker_original_steigend')};fail={};start=0;checkpoint=None
        if bootstrap:
            import json,hashlib
            from . import runner
            if runner.CHECKPOINT_ROOT is not None:
                h=hashlib.sha256()
                for a in (vx,dx,dy,design,y,ids,technical.subject_id.to_numpy(),development.subject_id.to_numpy()):
                    a=np.asarray(a);h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
                h.update(runner.sha(__file__).encode());h.update(str(runner.NUMERIC_BINDING).encode());h.update(json.dumps(parameters,sort_keys=True).encode())
                runner.CHECKPOINT_ROOT.mkdir(parents=True,exist_ok=True);checkpoint=runner.CHECKPOINT_ROOT/('ipw_'+h.hexdigest()+'.json')
                if checkpoint.exists():
                    state=json.loads(checkpoint.read_text(encoding='utf8'));vals=state['values'];fail=state['failed'];start=state['next_r']
        if bootstrap:
            for r in range(start,1000):
                if r%25==0:guard()
                try:
                    mult,rw,_=ipw_replicate(dx,dy,development.subject_id.to_numpy(),vx,technical.subject_id.to_numpy(),ids,r,parameters)
                    # ESS is over selected persons before outcome multiplicities;
                    # zero multiplicities remove persons, repeated draws retain IPW.
                    ess(mult*rw,y)
                    a=fit(design,y,ids,weights=mult,wald=False);b=fit(design,y,ids,weights=mult*rw,wald=False)
                    for k in ('beta','or','delta_auc','auc_G','auc_G_marker','auc_marker_original_steigend'):
                        vals[k].append(b[k]);vals.setdefault('unweighted_'+k,[]).append(a[k])
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:fail[str(exc)]=fail.get(str(exc),0)+1
                if checkpoint is not None and ((r+1)%25==0 or r==999):
                    checkpoint.write_text(json.dumps({'next_r':r+1,'values':vals,'failed':fail},allow_nan=False),encoding='utf8')
                    if (r+1)%100==0:print(json.dumps({'phase':'IPW_bootstrap','completed':r+1,'valid':len(vals['beta'])}),flush=True)
        weighted['bootstrap']={'planned':1000,'attempted':1000 if bootstrap else 0,'valid':len(vals['beta']),'failed':fail,
            'ci95':{k:np.quantile(v,[.025,.975]).tolist() if len(v)>=950 else None for k,v in vals.items()}}
        unweighted['bootstrap']={'planned':1000,'attempted':1000 if bootstrap else 0,'valid':len(vals['beta']),'failed':fail,
            'ci95':{k[len('unweighted_'):]:np.quantile(v,[.025,.975]).tolist() if len(v)>=950 else None for k,v in vals.items() if k.startswith('unweighted_')}}
        return {'status':'schaetzbar','unweighted_same_first_technical':unweighted,'IPW':weighted,'ESS_by_class':effective,
                'propensity_quantiles':np.quantile(prop,[0,.01,.5,.99,1]).tolist(),'weight_quantiles':np.quantile(w,[0,.01,.5,.99,1]).tolist(),
                'direction_reversed':weighted['beta']*unweighted['beta']<0,'missingness_fallfluss':flow,'fallen':FALLEN,
                'index_policy':'first technical adult; reference selection afterwards; no replacement','verification_flow':{'index_n':before,'complete_B4_without_marker':len(technical),'excluded_B4':before-len(technical)},'limitations':'MAR und ungemessene Indikation bleiben offen'}
    except (ValueError,KeyError,AttributeError,RuntimeError,np.linalg.LinAlgError) as exc:return stopped(exc)

def sensitivities(frames,package,union,technical=None,verification_train=None,bootstrap=True):
    h1=package['hypotheses']['BEF-0063'];f=frames['BEF-0063'];rv=frames['BEF-0087'];rv_entry=package['hypotheses']['BEF-0087'];out=[]
    def add(rank,name,fn):
        try:result=fn()
        except (ValueError,KeyError,AttributeError,RuntimeError,np.linalg.LinAlgError) as exc:result=stopped(exc)
        out.append({'rank':rank,'analysis':name,**result,'fallen':FALLEN})
    def est(selector=None,**kw):return estimate(f,h1,union,selector,bootstrap=bootstrap,**kw)
    def b4paired():
        _,good,_=masks(f,h1['endpoint'],h1['marker'],h1['comparator'],strong=True)
        return {'status':'berichtet','B4':estimate(f,h1,union,selector=lambda z:good,strong=True,bootstrap=bootstrap),
                'B3_same_set':est(lambda z:good),'paired_set_n':int(good.sum())}
    add('1','B4 und B3 auf identischem B4-vollständigem Set',b4paired)
    for bef,ep,marker,comp,direction in HYPOTHESES[1:]:
        def paired(bef=bef):
            entry=package['hypotheses'][bef];frame=frames[bef]
            _,good,_=masks(frame,entry['endpoint'],entry['marker'],entry['comparator'],strong=True)
            return {'status':'berichtet','B4':estimate(frame,entry,union,lambda z:good,strong=True,bootstrap=bootstrap),'B3_same_set':estimate(frame,entry,union,lambda z:good,bootstrap=bootstrap),'paired_set_n':int(good.sum())}
        add('1','B4 und B3 auf identischem Set / '+bef,paired)
    add('1b','Grenzsensitivität Echo <=29 Tage',lambda:est(lambda z:z.time_delta_hours.abs().le(696)))
    add('2a','Echo ±7 Tage am unveränderten Index',lambda:est(lambda z:z.time_delta_hours.abs().le(168)))
    add('2b','LVEF <=40 gegen >40 auf primärem Fallset',lambda:est(inclusive=True))
    add('3','IPW / ungegewichtet auf demselben ersten technischen Index',lambda:ipw_analysis(verification_train,technical,h1,package['verification'],union,bootstrap) if technical is not None and verification_train is not None else stopped('attestiertes erstes technisches Indexuniversum fehlt'))
    add('3b','Verifikationsassoziation Marker+B4 und Alter/Geschlecht/QRS',lambda:verification_associations(technical,h1,package['variants']['verification_marker'],union,bootstrap) if technical is not None else stopped('erstes technisches Indexuniversum fehlt'))
    add('4','LVEF <=40 gegen LVH positiv/LVEF >=50, identisches TTE',lambda:est(lvh_selector,inclusive=True))
    add('4_LSB','LVH bei LSB: Entwicklungs-Supplement',lambda:stopped('AE-103: nicht schätzbarer Entwicklungsbefund; kein Testersatz'))
    add('5a','RV bei LVEF >=50, DTW',lambda:estimate(rv,rv_entry,union,rv_selector,bootstrap=bootstrap))
    def rvfrechet():
        params=package['variants']['RV_frechet']
        return estimate(rv,rv_entry,union,rv_selector,marker=h1['marker'],parameter=params,bootstrap=bootstrap)
    add('5b','RV bei LVEF >=50, Fréchet deskriptiv',rvfrechet)
    add('6a','QRS <120 ms, ohne positive Pipeline-LBBB/RBBB-Muster',lambda:est(lambda z:z.qrs_duration_ms.lt(120)&z.classical_lbbb_pattern_positive.eq(False)&z.classical_rbbb_pattern_positive.eq(False)))
    add('6b','Sinusselektor nach AW054-Regex',lambda:est(lambda z:truth(z.sinus_selector)))
    for group in ('lt65','ge65','F','M'):
        add('6c' if '65' in group else '6d',group,lambda group=group:est(lambda z:z.age.lt(65) if group=='lt65' else z.age.ge(65) if group=='ge65' else z.sex.eq(group),sex_stratum=group if group in ('F','M') else None))
    def ntcontrol():
        # Add the frozen extra numeric regressor immediately before the marker.
        return estimate_extra(f,h1,union,nt_numeric(f),package['variants']['H1_NT'],bootstrap)
    add('6e','Gemeinsame Echo-/NT-Referenz + log1p(NT-proBNP/URL)',ntcontrol)
    add('6e_RV','RV + log1p(NT-proBNP/URL)',lambda:estimate_extra(rv,rv_entry,union,nt_numeric(rv),package['variants']['RV_NT'],bootstrap))
    for bef,ep,marker,comp,direction in HYPOTHESES[1:]:
        for rank,name,selector in [
            ('6a','QRS <120 ms, ohne positive Pipeline-LBBB/RBBB-Muster',lambda z:z.qrs_duration_ms.lt(120)&z.classical_lbbb_pattern_positive.eq(False)&z.classical_rbbb_pattern_positive.eq(False)),
            ('6b','Sinusselektor nach AW054-Regex',lambda z:truth(z.sinus_selector)),
            ('6c','Alter <65',lambda z:z.age.lt(65)),('6c','Alter >=65',lambda z:z.age.ge(65)),('6d','Geschlecht F',lambda z:z.sex.eq('F')),('6d','Geschlecht M',lambda z:z.sex.eq('M'))]:
            add(rank,name+' / '+bef,lambda bef=bef,selector=selector,rank=rank,name=name:estimate(frames[bef],package['hypotheses'][bef],union,selector,bootstrap=bootstrap,sex_stratum=name[-1] if rank=='6d' else None))
    signal_columns=['original','dower','direct8_centered_rms','direct8_qrs_max_centered','kors_qrs_max_centered','shape','residual']
    for rank,name,cols in [('7','Inverse Dower',['original','dower']),('8a','direct8 centered-RMS',['original','direct8_centered_rms']),('8b','QRS-max centered',['original','direct8_qrs_max_centered','kors_qrs_max_centered']),('9','Original / train-fixes Residuum / SO(3)-Formrest',['original','residual','shape'])]:
        add(rank,name,lambda:stopped('nicht schätzbar (Skalierung nicht gebunden)'))
    return out

def estimate_extra(f,entry,union,extra,params,bootstrap=True):
    from .runner import bootstrap as resample
    try:
        z,good,flow=masks(f,entry['endpoint'],entry['marker'],entry['comparator']);good &= np.isfinite(extra)
        x,names=raw_design(z.loc[good],entry['marker'],entry['comparator']);x=np.column_stack([x[:,:-1],extra[good],x[:,-1]])
        if params['names']!=names[:-1]+['nt_log_url']+names[-1:]:raise ValueError('NT-Design nicht gebunden')
        design=fixed_design(x,params);ids=z.loc[good,'subject_id'].to_numpy();y=classify_lvef(z.loc[good,'preferred_lvef']) if entry['endpoint']=='reduced_lvef_strict' else z.loc[good,'label_state'].eq('positive').to_numpy(float)
        result=fit(design,y,ids);result.update(status='schaetzbar',fallen=FALLEN,missingness_fallfluss=flow)
        if bootstrap:result['bootstrap']=resample(design,y,ids,union)
        return result
    except (ValueError,KeyError,RuntimeError,np.linalg.LinAlgError) as exc:return stopped(exc)

def run_frames(frames,package,technical=None,verification_train=None,development=False,bootstrap=True):
    union=np.unique(np.concatenate([f.subject_id.to_numpy() for f in frames.values()]+([] if technical is None else [technical.subject_id.to_numpy()])))
    results={}
    for bef,ep,marker,comp,direction in HYPOTHESES:
        import json
        print(json.dumps({'phase':'association_fit','endpoint':bef,'role':'entwicklung' if development else 'test'}),flush=True)
        entry=package['hypotheses'][bef]
        entry['_require_measurement']=not development
        if not development and entry['measurement_attestation'].get('status')!='bestanden':results[bef]=stopped(entry['measurement_attestation'].get('reason','Messattest fehlt'))
        else:results[bef]=estimate(frames[bef],entry,union,bootstrap=bootstrap)
        if development:results[bef]['claim']='Entwicklung; Messgates nicht als unabhängiger Test bestanden'
    h1=results['BEF-0063'].get('urteil')=='erfolg'
    for bef,v in holm(results,h1).items():
        results[bef]['multiplicity']=v
        if results[bef]['status']=='schaetzbar':results[bef]['urteil']='erfolg' if v['confirmatory_success'] else 'statistischer_misserfolg'
    if development or package['hypotheses']['BEF-0063']['measurement_attestation'].get('status')=='bestanden':
        controls=sensitivities(frames,package,union,technical,verification_train,bootstrap)
    else:
        from .runner import SENSITIVITIES
        controls=[{'rank':rank,'analysis':name,**stopped('nicht schätzbar (Skalierung nicht gebunden)' if rank in ('7','8a','8b','9') else 'H1-Mess-/Quellenattest fehlt; keine Ersatzanalyse')} for rank,name,_ in SENSITIVITIES]
    return {'role':'entwicklung' if development else 'test','partition':'validation' if development else 'test','H1_gate':h1,'results':results,
            'sensitivities':controls,'calibration':'nicht konfirmatorisch prüfbar unter K1'}
