"""Entwicklungsparameter für fest registrierte Zusatzspalten und Pilotvarianten."""
import json
import hashlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .export import OLD,OUT,BASE,projected,save
from .daten import masks
from .statistik import raw_design,scaling
from .adapter import residual,nt_numeric,verification_design
from werkzeuge.b_vorbereitung.clinical import reference
from werkzeuge.b_vorbereitung.runner import sha

def export_extras(package):
    binding=json.loads((OLD/'clinical_binding.json').read_text(encoding='utf8'))
    for name in ('echo_evidence_private.parquet','clinical_first_technical_private.parquet'):
        if sha(OLD/name)!=binding['projections_sha256'][name]:raise ValueError('Zusatzinputhash verändert: '+name)
    primary=package['hypotheses']['BEF-0063'];f=projected(OUT/'BEF-0063_entwicklung.parquet')
    e=projected(OLD/'echo_evidence_private.parquet')
    variants={}
    def parameters(frame,entry,marker,filter_text,source,extra=None):
        try:
            z,good,flow=masks(frame,entry['endpoint'],marker,entry['comparator'])
            if extra is not None:good &= np.isfinite(extra)
            x,names=raw_design(z.loc[good],marker,entry['comparator'])
            if extra is not None:x=np.column_stack([x[:,:-1],extra[good],x[:,-1]]);names=names[:-1]+['nt_log_url']+names[-1:]
            p=scaling(x,names);p['status']='gebunden'
            # Alle unveränderten B3-Regressoren und der Originalmarker behalten
            # das primäre SD-Maß. Nur neue Variantenspalten werden exportiert.
            base=entry['scaling_B3']
            for i,n in enumerate(names):
                original=entry['marker'] if n=='original' else n
                if original in base.get('names',[]):
                    j=base['names'].index(original);p['mean'][i]=base['mean'][j];p['sd'][i]=base['sd'][j];p['keep'][i]=base['keep'][j]
        except (ValueError,KeyError,AttributeError) as exc:p={'status':'nicht_schaetzbar','reason':str(exc)}
        p['origin']={'run':'A155-2026-10-04-d6e70e','file':str(source),'sha256':sha(source),'filter':filter_text,'script_sha256':sha(__file__)}
        p['binding_sha256']=hashlib.sha256(json.dumps(p,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        return p
    rv_entry=package['hypotheses']['BEF-0087'];rv=projected(OUT/'BEF-0087_entwicklung.parquet')
    variants['RV_frechet']=parameters(rv,rv_entry,primary['marker'],'train+validation RV-B3 complete; alternative Fréchet',OLD/'echo_rv_systolic_dysfunction_private.parquet')
    try:nt=nt_numeric(f)
    except (ValueError,AttributeError):nt=pd.Series(np.nan,index=f.index)
    variants['H1_NT']=parameters(f,primary,primary['marker'],'train+validation; H1 + exact uncensored NT/URL; hour alignment true',OLD/'reduced_lvef_strict_private.parquet',nt)
    try:nt_rv=nt_numeric(rv)
    except (ValueError,AttributeError):nt_rv=pd.Series(np.nan,index=rv.index)
    variants['RV_NT']=parameters(rv,rv_entry,rv_entry['marker'],'train+validation; RV + exact uncensored NT/URL; hour alignment true',OLD/'echo_rv_systolic_dysfunction_private.parquet',nt_rv)
    # Keine Erweiterung der Entwicklungssignalrechnung: nur vorhandene AW121-Werte.
    vp=OLD/'signal_values_private.json';values=json.loads(vp.read_text(encoding='utf8'))
    if values['request_sha256']!=sha(OLD/'signal_request_private.json'):raise ValueError('Signalrequesthash falsch')
    v=pd.DataFrame(values['rows']);paired=f.merge(v,on=['subject_id','study_id','split'],validate='one_to_one')
    cols=['original','dower','direct8_centered_rms','direct8_qrs_max_centered','kors_qrs_max_centered','shape']
    common=np.isfinite(paired[cols]).all(axis=1)&paired.equivalence.eq(True);paired=paired.loc[common].copy()
    paired['residual']=residual(paired.original,paired[['qrs_duration_ms','qt_interval_ms','context_heart_rate_bpm','age']],paired.sex,package['residual'])
    for c in cols+['residual']:
        variants[c]=parameters(paired,primary,c,'AW121 gemeinsames Signal-Pilotfallset train+validation; keine Vollpopulations-Erweiterung',vp)
        variants[c]['limitation']='Neue Varianten-SD ausschließlich aus gebundenem Entwicklungs-Signalpilot; K muss dies vor Test bestätigen (§5/§9-Populationsvertrag). Kein austauschbares Hauptmarker-SD-Maß.'
        if c!='original':variants[c]['status']='gebunden_pilot_STOP_Vollpopulationsskalierung'
    # Varianten-SD werden nur bei vor Ergebnissicht bestätigter Pilotbindung
    # testseitig verwendet; Originalmarker bleibt auf dem Haupt-SD-Maß.
    for c in cols+['residual']:
        variants[c]['binding_sha256']=hashlib.sha256(json.dumps({k:v for k,v in variants[c].items() if k!='binding_sha256'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    # Gemeinsame Echo-Referenzen ergänzen; keine Neuklassifikation/Indexwahl.
    for bef,ep in [('BEF-0063','echo_lvh_strict'),('BEF-0087','reduced_lvef_strict')]:
        entry=package['hypotheses'][bef];frame=f if bef=='BEF-0063' else rv
        frame=reference(frame,e,ep)
        p=OUT/f'{bef}_entwicklung.parquet';pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),p,row_group_size=4096)
        entry['input_development']={'path':str(p),'sha256':sha(p)}
        entry['additional_reference_origin']={'file':str(OLD/'echo_evidence_private.parquet'),'sha256':sha(OLD/'echo_evidence_private.parquet'),'join':'study_id; one_to_one; same TTE checked by eligible'}
    # Attestiertes erstes technisches Entwicklungsuniversum und Outcome-Evidence.
    tech_source=OLD/'clinical_first_technical_private.parquet'
    tech=reference(projected(tech_source),e,'reduced_lvef_strict')
    prefix='reduced_lvef_strict__'
    for c in ['label_state','eligibility_state','time_delta_hours','time_precision','evidence_ids_json','ehr_time_alignment_confirmatory_eligible']:
        tech[c]=tech[prefix+c]
    tech['index_selection_strategy']='first_technical_adult'
    # preindex/clinical_first_technical already carries recorded sex and the
    # gated machine heart rate. Do not invent a positive metadata attestation.
    if 'machine_metadata_usable_for_primary' not in tech:
        pre=projected(OLD/'preindex.parquet',['subject_id','study_id','split','sex','heart_rate'])
        # heart_rate originates from CASE WHEN machine_metadata_usable...;
        # only finite gated values can contribute to this development fit.
        if 'sex' not in tech:tech=tech.merge(pre[['subject_id','study_id','split','sex']],on=['subject_id','study_id','split'],validate='one_to_one')
    x,names=raw_design(tech,primary['marker'],strong=True);complete=np.isfinite(x).all(axis=1)
    vp=scaling(x[complete],names);vp['status']='gebunden'
    vp['mean'][-1]=primary['scaling_B3']['mean'][-1];vp['sd'][-1]=primary['scaling_B3']['sd'][-1]
    vp['origin']={'file':str(tech_source),'sha256':sha(tech_source),'filter':'train+validation; first technical adult; complete B4+marker; reference availability not reference result'}
    variants['verification_marker']=vp
    tp=OUT/'first_technical_entwicklung.parquet';pq.write_table(pa.Table.from_pandas(tech,preserve_index=False),tp,row_group_size=4096)
    # Fehlende B4-Eingänge im Verifikationsuniversum werden separat gezählt.
    train=tech.loc[tech.split.eq('train')].copy();x,_=raw_design(train.assign(__verification=0.),'__verification',strong=True)
    train=train.loc[np.isfinite(x[:,:-1]).all(axis=1)].copy()
    train_path=OUT/'verification_train.parquet';pq.write_table(pa.Table.from_pandas(train,preserve_index=False),train_path,row_group_size=4096)
    package['technical_development']={'path':str(tp),'sha256':sha(tp),'origin':{'file':str(tech_source),'sha256':sha(tech_source),'filter':'first technical adult; train+validation; reference join by study_id'}}
    package['verification_train']={'path':str(train_path),'sha256':sha(train_path),'n':len(train),'filter':'first technical adult; train; complete B4 WITHOUT marker'}
    package['variants']=variants
    return package
