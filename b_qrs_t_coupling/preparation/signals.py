"""Deterministischer Signalprobelauf auf vorhandener Entwicklungsstichprobe.

200 Personen je Fold aus AW023b, nach Identitätshash (keine neue Outcome-Wahl).
Gleiche 400 angefragte EKGs für alle Transformationen/Zerlegungen; gemeinsames
Complete-case-Set für alle Fits. Keine Übertragung dieser Stichprobe auf den Vollnenner.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json,hashlib,time
import numpy as np
import pandas as pd
from .runner import small,sha,save,PROJECT
from .models import MARKER,BASE,EXTRA,compare
from .clinical import reference,eligible,complete
from .kernels import coupling,LEADS

def run(out):
    assert json.loads((out/'synthetic.json').read_text(encoding='utf8'))['status']=='passed'
    from werkzeuge.varianten.adapter import prepare_record,context
    from werkzeuge.varianten.metriken import form_distance
    from werkzeuge.varianten.residuen import Residualizer
    from werkzeuge.nachrechnung.resources import available
    from werkzeuge.varianten.adapter import release_import
    release_import()
    from vcg_biomarker.core.biomarkers.discovery_v3_biomarkers import _qrs_t_distances
    contextp=__import__('pathlib').Path(str(data_path("b.varianten_aw023c_stichprobe_kontext_parquet")))
    sourcebind=json.loads(contextp.with_name('kontextbericht.json').read_text(encoding='utf8'))
    assert sha(contextp)==sourcebind['context_sha256']
    ctx=small(contextp);f=small(out/'reduced_lvef_strict_private.parquet');e=small(out/'echo_evidence_private.parquet');f=reference(f,e,'reduced_lvef_strict')
    good=eligible(f,'reduced_lvef_strict')&complete(f,MARKER)&np.isfinite(f.qt_interval_ms)&np.isfinite(f.context_heart_rate_bpm)&~f.lvef_conflict_gt_10.fillna(True)
    f=f[good].merge(ctx[['subject_id','study_id','split']],on=['subject_id','study_id','split'],validate='one_to_one')
    f['rank']=f.apply(lambda r:hashlib.sha256(f'121|{r.split}|{r.subject_id}|{r.study_id}'.encode()).hexdigest(),axis=1)
    selected=pd.concat([f[f.split.eq(s)].sort_values('rank').head(200) for s in ('train','validation')]).drop(columns='rank')
    assert not selected.subject_id.duplicated().any()
    request=selected[['subject_id','study_id','split']].to_dict('records');bind={'context_sha256':sha(contextp),'clinical_sha256':sha(out/'reduced_lvef_strict_private.parquet'),'selected':request,'sample_design':'200 je Fold, hash121; bestehende AW023b-Stichprobe outcome-/QRS-/Rhythmus-stratifiziert; keine repräsentative Prävalenz','code_sha256':sha(__file__)}
    save(out/'signal_request_private.json',bind)
    # DataFrame missing values -> None, no patient IDs in stdout/log.
    records=selected[['subject_id','study_id','split']].merge(ctx,on=['subject_id','study_id','split'],validate='one_to_one');records=records.astype(object).where(records.notna(),None).to_dict('records')
    expected=selected.set_index('study_id')[MARKER].to_dict();values=[];path=out/'signal_values_private.json';start=time.perf_counter()
    if path.exists():
        old=json.loads(path.read_text(encoding='utf8'));assert old['request_sha256']==sha(out/'signal_request_private.json');values=old['rows']
    for i,row in enumerate(records):
        if i<len(values):continue
        available();rec={k:row[k] for k in ('subject_id','study_id','split')}
        # Existing adapter has a legacy per-record guard; this assignment affects
        # only this authorized local process, not source files or the analysis job.
        import werkzeuge.varianten.adapter as adapter
        adapter.memory_guard=available
        try:
            package=prepare_record(row);c=context(package);d=context(package,'inverse_dower')
            original=_qrs_t_distances(c);dower=_qrs_t_distances(d)
            if original is None or not np.isclose(original['normalized_discrete_frechet'],expected[row['study_id']],rtol=1e-8,atol=0):
                raise ValueError('Originaläquivalenzgate')
            q=c.vcg_segment_origin_corrected('qrs');t=c.vcg_segment_origin_corrected('t')
            q8=np.column_stack([c.ecg_segment(l,'qrs') for l in LEADS]);t8=np.column_stack([c.ecg_segment(l,'t') for l in LEADS])
            rec.update({'original':original['normalized_discrete_frechet'],'dower':None if dower is None else dower['normalized_discrete_frechet'],
              'direct8_centered_rms':coupling(q8,t8),'direct8_qrs_max_centered':coupling(q8,t8,'qrs_max'),
              'kors_qrs_max_centered':coupling(q,t,'qrs_max'),
              'shape':form_distance(c.vcg_segment('qrs'),c.vcg_segment('t'),'rotation'),'fs':float(package['fs']),
              'equivalence':True,'abstention':''})
            # Generic centered-RMS kernel against actual Release at each valid EKG.
            if not np.isclose(coupling(q,t),rec['original'],rtol=1e-8,atol=0):raise ValueError('Kernäquivalenzgate')
            del package,c,d
        except (ValueError,TypeError,IndexError) as exc:rec.update({'equivalence':False,'abstention':str(exc)})
        values.append(rec)
        if (i+1)%10==0 or i+1==len(records):
            save(path,{'request_sha256':sha(out/'signal_request_private.json'),'rows':values})
            print(json.dumps({'phase':'signals','completed':i+1,'requested':len(records),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    v=pd.DataFrame(values);cols=['original','dower','direct8_centered_rms','direct8_qrs_max_centered','kors_qrs_max_centered','shape']
    for name in cols:
        if name not in v:v[name]=np.nan
    matched=selected.merge(v,on=['subject_id','study_id','split'],validate='one_to_one')
    paired=matched[np.isfinite(matched[cols]).all(axis=1)&matched.equivalence.eq(True)].copy()
    numeric=['qrs_duration_ms','qt_interval_ms','context_heart_rate_bpm','age'];tr=paired.split.eq('train')
    if min(tr.sum(),(~tr).sum())<20:
        save(out/'signals_result.json',{'status':'nicht_schaetzbar','requested':len(records),'common_n':len(paired),'reason':'gemeinsames Kernäquivalenz-/Fallset unzureichend','abstentions':v.abstention.value_counts().to_dict()});return
    residual=Residualizer().fit(paired.loc[tr,numeric],paired.loc[tr,'sex'],paired.loc[tr,'original'])
    paired['residual']=residual.residual(paired[numeric],paired.sex,paired.original)
    save(out/'residual_fit_frozen.json',{'recipe':'VB-VF01-08 unverändert: Huber epsilon1.35 alpha0 max_iter200 tol1e-5, QRS/QT/HF/Alter, kategoriales Geschlecht, SD ddof1; nur train',
      'mean':residual.mean.tolist(),'sd':residual.sd.tolist(),'categories':residual.categories,'coef':residual.model.coef_.tolist(),'intercept':float(residual.model.intercept_),
      'recipe_sha256':sha(PROJECT/'werkzeuge/varianten/residuen.py'),'shape_recipe_sha256':sha(PROJECT/'werkzeuge/varianten/metriken.py')})
    y=paired.preferred_lvef.lt(40).to_numpy(float);result={name:compare(paired,y,paired[name].to_numpy(float)) for name in cols+['residual']}
    result['_flow']={'requested':len(records),'by_fold_requested':selected.split.value_counts().to_dict(),'common_n':len(paired),'abstentions':v.abstention.value_counts().to_dict(),'sampling':'Entwicklungsprobelauf, bestehende stratifizierte Stichprobe; kein Vollnenner','normalization_conflict':'QRS-max-centered technisch zusätzlich; normative Formulierung §9.5 widerspricht Release-centered-RMS, keine Gleichsetzung','fs':sorted(v.fs.dropna().unique().tolist()) if 'fs' in v else []}
    save(out/'signals_result.json',result)
    print(json.dumps({'phase':'signals_finished','common_n':len(paired),'results':{k:{a:z.get(a) for a in ('status','delta_auc','or_per_train_sd')} for k,z in result.items() if not k.startswith('_')}}),flush=True)
