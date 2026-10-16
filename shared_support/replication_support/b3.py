"""AW-034: additive B3-Nachrechnung, ausschließlich Entwicklungsdaten."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import argparse, csv, gc, json, re, time
from pathlib import Path
from datetime import datetime, timezone
from . import runner as r
from .resources import guard, Budget

OUT=Path(str(data_path("b.nachrechnung_b3")))
OLD=Path(str(data_path("b.nachrechnung")))
ANGLE='spatial_qrs_t_angle_deg'
SPEC=r.PROJECT/'nachrechnung/SPEZIFIKATION_B3_NACHTRAG.md'

def now():return datetime.now(timezone.utc).isoformat()

def prepare():
    import yaml
    family_files=sorted((r.PROJECT/'befunde/_familien').glob('F1_*'))+[r.PROJECT/'koordination/WELLE4_BERICHT.md']
    family=set()
    for p in family_files:family.update(re.findall(r'BEF-\d{4}',p.read_text(encoding='utf-8')))
    rows=list(csv.DictReader((r.PROJECT/'nachrechnung/ergebnis_n0_n2_n4.csv').open(encoding='utf-8-sig'),delimiter=';'))
    previous={x['id']:x for x in rows}
    items=[];excluded=[]
    for p in sorted((r.PROJECT/'befunde').glob('*/befund.yaml')):
        b=yaml.safe_load(p.read_text(encoding='utf-8'));ident=b['id'];old=previous.get(ident,{})
        keys=[q.get('schluessel',{}) for q in b['herkunft']['quellen']]
        key=next((k for k in keys if all(c in k for c in ('endpoint_id','panel_id','feature_id'))),None)
        reasons=[]
        if ident in family or (key and ('qrs_t_coupling' in p.parent.name or key['feature_id'].startswith('qrs_t_'))):reasons.append('a: F1/QRS-T-Kopplung')
        if old.get('status')=='completed' and b['ergebnistyp']=='Q' and (old.get('B2_identisch_B1')=='True' or b['klinikfeld']=='herzinsuffizienz_struktur'):reasons.append('b: AW033 abgeschlossen/B2=B1 oder Herzinsuffizienz-Struktur')
        if not reasons:continue
        item={'id':ident,'typ':b['ergebnistyp'],'klinikfeld':b['klinikfeld'],'key':key,'reasons':reasons,'akte':str(p),'akte_sha256':r.sha(p),'fallen_akte':b.get('fallen',{}),'AW033_status':old.get('status','nicht_erfasst')}
        if b['status']=='verworfen':excluded.append(item);continue
        marker=key['feature_id'] if key else ''
        duplicate=marker in (ANGLE,'qrs_area_to_t_area_angle_deg','spatial_qrs_t_angle_deg_catalog','qrs_area_to_t_area_angle_deg_axis_relation','spatial_qrs_t_angle_mean')
        item['pre_status']='nicht anwendbar, Duplikat' if duplicate else ('berechenbar' if old.get('status')=='completed' and b['ergebnistyp']=='Q' else 'Vertragsausnahme: '+item['AW033_status'])
        items.append(item)
    inventory={'created_at':now(),'items':items,'excluded_verworfen':excluded,'sources_sha256':{str(p):r.sha(p) for p in family_files+[r.PROJECT/'nachrechnung/ergebnis_n0_n2_n4.csv']}}
    if SPEC.exists():raise ValueError('Vorabnachtrag existiert bereits; nicht neu einfrieren')
    text=['# AW-034 - Vorabnachtrag B3','',f'Datiert und eingefroren: {inventory["created_at"]}. Vor individueller B3-Rechnung. Post-hoc Auswertungsplan, keine Studienpräregistrierung.',
      '', 'Basis unverändert: SPEZIFIKATION_N0_N2_N4.md, SHA-256 '+r.sha(r.PROJECT/'nachrechnung/SPEZIFIKATION_N0_N2_N4.md')+'.',
      '', 'B3 = B2 + spatial_qrs_t_angle_deg, linear; Pipeline-Kors-Merkmal aus exakt derselben Entwicklungs-Featurepartition. Winkel = arccos des normierten Skalarprodukts origin-korrigierter QRS- und vollständiger JT-Flächenvektoren in Grad. QRS: onset bis offset einschließlich; JT: QRS-offset bis T-offset einschließlich ST. Trapezintegration. Release core/vcg_geometry.py:191-229; core/biomarkers/catalog_biomarkers.py:1882-1926. Keine Neuberechnung/Ersetzung des Pipeline-Merkmals.',
      '', 'Unverändert: Folds A train→validation/B validation→train, attestierter Index, eine Person je Endpunkt, B0/B1/B2, Spline, Standardisierung, Ridge, Newton-Konvergenz, keine Imputation/Optimierung. Neue gemeinsame vollständige Fälle über B0/B1/B2/B3/Marker; alle B0-B3-Modelle ohne/mit Marker neu auf diesen Fällen fitten. 1.000 gemeinsame Patientencluster-Replikate innerhalb Originalfolds, Seed 32009+BEF-Nummer+1000; vollständiger Neufit, 95%-Perzentil-KI, feste Fallzahl-Poolgewichte und Foldkovarianz. Maximal 2.000 Versuche, nur Ein-Klassen-Replikate nachziehen. Bedingte AUC weiterhin gehaltenes Residuum mit train-gelernter Richtung; keine kausale conditional ROC.',
      '', 'Zusätzlich: Spearman Marker↔Winkel auf verfügbaren Entwicklungs-Indexfällen und auf neuer gemeinsamer Fallmenge, separat train/validation und gepoolt; Fallzahlen dazu. Winkel allein: rohe und gehalten gerichtete AUC auf neuer gemeinsamer Fallmenge, Richtung ausschließlich Gegenfold; KI aus denselben 1.000 Replikaten. Rohmarker-AUC alte/neue Fallmenge, Fallfluss und Fehlwerte pro Klasse berichten. Keine neue Wertung, Schwelle, Auswahl oder klinische Handlungsempfehlung.',
      '', 'Hashgeprüfte vollständige Entwicklungsprojektionen und N0-Gates aus AW033/AW032 weiterverwenden, Winkel mit Arrow-ID-Filter aus Originalpartitionen ergänzen; historische Ausgangsbasis B0-B2 separat erhalten. Nicht abgeschlossene Familienakten werden als Vertragsausnahme mit ursprünglichem Grund dokumentiert; keine fremde Indexstrategie oder fehlenden B2-Komparator ersetzen. Winkel/Aliasse algebraische Duplikate nicht fitten. Gesperrte Testmenge niemals öffnen.',
      '', 'Ein Python-Prozess ≤1,5 GiB; vor jedem Feature-/25-Replikateblock RAM und Commit ≥4 GiB, sonst warten. DuckDB falls verwendet 1 GB/2 Threads; numerische Threads 1. Kontrollpunkte je Befund unter <local path>; RNG und Spezifikations-/Code-/Eingangshashes binden Fortsetzung.',
      '', '## Befundliste vor Rechnung','', '| Befund | Endpunkt | Marker | Auswahl | Vorabstatus |','|---|---|---|---|---|']
    for x in items:text.append(f'| {x["id"]} | {(x["key"] or {}).get("endpoint_id", "")} | {(x["key"] or {}).get("feature_id", "")} | {"; ".join(x["reasons"])} | {x["pre_status"]} |')
    text+=['','Ausgeschlossene verworfene Akten: '+', '.join(x['id'] for x in excluded)+'.','', 'Vollständige Auswahl-/Akten-/Quellenbindung: inventory_AW034.json (E:-Ausgabe).']
    SPEC.write_text('\n'.join(text)+'\n',encoding='utf-8')
    inventory['spec_sha256']=r.sha(SPEC);r.write(OUT/'inventory_AW034.json',inventory)
    print(json.dumps({'phase':'frozen','sha256':inventory['spec_sha256'],'items':len(items),'statuses':{s:sum(x['pre_status']==s for x in items) for s in set(x['pre_status'] for x in items)}}),flush=True)

def verified_old(item):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .core import assert_development
    ident=item['id'];path=OLD/ident/(ident+'.json');old=json.loads(path.read_text(encoding='utf-8'))
    assert r.sha(path)==json.loads(path.with_suffix('.sha256.json').read_text(encoding='utf-8'))['sha256']
    assert old['N0']['status']=='passed' and old['N2']['status']=='completed'
    b=old['input_binding']
    for p,h in b['files_sha256'].items():assert r.sha(p)==h,p
    assert r.sha(r.DATA/'labels/development_label_attestation.json')==b['label_attestation_sha256']
    assert r.sha(r.DATA/'labels/endpoint_index_cohorts.parquet')==b['labels_sha256']
    manifest=json.loads((path.parent/'input_manifest.json').read_text(encoding='utf-8'))
    for part in manifest['partition_manifest']:
        source=Path(part['path'])
        assert source.stat().st_size==part['size'] and source.stat().st_mtime_ns==part['mtime_ns'],str(source)
    projection=path.parent/'development_projection.parquet';s=pq.ParquetFile(projection).schema_arrow
    t=pq.read_table(projection,columns=s.names)
    f=t.to_pandas()
    # Gleiche Pandas→Arrow-Repräsentation wie Erzeuger und AW033-Abschlussaudit;
    # Parquet normalisiert physische Werte unter Nullmasken.
    assert r.arrow_sha(pa.Table.from_pandas(f,preserve_index=False))==b['cohort_feature_projection_sha256']
    assert_development(f)
    assert not f.subject_id.duplicated().any() and not f.study_id.duplicated().any()
    return f,old,{'old_result_sha256':r.sha(path),'old_input_binding':b,'projection_file_sha256':r.sha(projection)}

def angles(items):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    pa.set_cpu_count(2);pa.set_io_thread_count(2)
    needed=[]
    for x in items:
        if x['pre_status']=='berechenbar':
            f,_,_=verified_old(x);needed.append(f[['subject_id','study_id','split']]);del f;gc.collect()
    wanted=pd.concat(needed).drop_duplicates().sort_values(['split','study_id'])
    assert not wanted.study_id.duplicated().any()
    rows=[];manifest=[]
    for split in ('train','validation'):
        ids=wanted.loc[wanted.split==split,'study_id'].tolist()
        for i,p in enumerate(sorted((r.DEV/f'split={split}').glob('part-*.parquet'))):
            guard();pf=pq.ParquetFile(p);s=pf.schema_arrow
            columns=['subject_id','study_id','split']+([ANGLE] if ANGLE in s.names else [])
            t=ds.dataset(str(p),format='parquet').to_table(columns=columns,filter=ds.field('study_id').isin(ids))
            with p.open('rb') as stream:
                stream.seek(-8,2);footerlen=int.from_bytes(stream.read(4),'little');stream.seek(-8-footerlen,2);footer=stream.read(footerlen+8)
            import hashlib
            manifest.append({'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'footer_sha256':hashlib.sha256(footer).hexdigest(),'projection_sha256':r.arrow_sha(t),'selected_rows':t.num_rows,'columns':columns})
            if t.num_rows:
                f=t.to_pandas();assert set(f.split)=={split}
                if ANGLE not in f:f[ANGLE]=np.nan
                rows.append(f)
            if i%200==0:print(json.dumps({'phase':'angle_features','split':split,'files':i}),flush=True)
    a=pd.concat(rows,ignore_index=True);assert not a.study_id.duplicated().any()
    assert len(a.merge(wanted,on=['subject_id','study_id','split']))==len(a)
    a=wanted.merge(a,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
    a[ANGLE]=pd.to_numeric(a[ANGLE],errors='coerce').replace([np.inf,-np.inf],np.nan)
    path=OUT/'angles_development.parquet';pq.write_table(pa.Table.from_pandas(a,preserve_index=False),path)
    r.write(OUT/'angles_manifest.json',{'partitions':manifest,'projection_file_sha256':r.sha(path),'sealed_read':False})
    return a

def bootstrap(ident,binding,one):
    import numpy as np
    path=OUT/ident/'B3_checkpoint.json';signature=r.digest(binding);seed=32009+int(ident[4:])+1000
    rng=np.random.default_rng(seed);values=[];attempts=0;seconds=0
    if path.exists():
        cp=json.loads(path.read_text(encoding='utf-8'));assert cp['signature']==signature,'Checkpointbindung'
        values=cp['values'];attempts=cp['attempts'];seconds=cp['seconds'];rng.bit_generator.state=cp['rng_state']
    last=time.perf_counter()
    while len(values)<1000:
        if len(values)%25==0:guard()
        attempts+=1
        if attempts>2000:raise RuntimeError('Zu viele ungültige Replikate')
        try:values.append(one(rng).tolist())
        except ValueError as e:
            if str(e)!='Nur eine Klasse':raise
            continue
        if len(values)%25==0:
            tick=time.perf_counter();seconds+=tick-last;last=tick
            r.write(path,{'signature':signature,'values':values,'rng_state':rng.bit_generator.state,'attempts':attempts,'seconds':seconds,'seed':seed})
            print(json.dumps({'befund':ident,'phase':'B3','valid_replicates':len(values),'seconds':round(seconds,1)}),flush=True)
    return np.asarray(values),{'iterations':1000,'attempts':attempts,'seed':seed,'seconds':seconds}

def calculate(frame,old,item,binding):
    import numpy as np
    from scipy.stats import spearmanr
    from .core import design,held_models,auc,cluster_weights,pool_summary
    marker=item['key']['feature_id'];comp=old['input_binding']['comparator'];cf=comp['feature_used']
    required=list(dict.fromkeys(['age','sex_numeric','heart_rate','qrs_duration_ms','qtc_fridericia_ms',cf,ANGLE,marker]))
    observed=frame.loc[frame[marker].notna()].copy();complete=observed.dropna(subset=required).copy()
    splits=[complete.loc[complete.split==s] for s in ('validation','train')]
    arrays={m:[design(s,m,cf) for s in splits] for m in ('B0','B1','B2')}
    arrays['B3']=[np.column_stack([arrays['B2'][i],s[ANGLE].to_numpy(float)]) for i,s in enumerate(splits)]
    labels=[s.label.to_numpy() for s in splits];markers=[s[marker].to_numpy(float) for s in splits];angle=[s[ANGLE].to_numpy(float) for s in splits]
    def calc(w):
        results=[]
        for m in ('B0','B1','B2','B3'):
            if m=='B2' and comp['B2_equals_B1']:results.append(results[-1].copy());continue
            results.append(np.array([held_models(arrays[m][1-i],arrays[m][i],markers[1-i],markers[i],labels[1-i],labels[i],w[1-i],w[i]) for i in range(2)]))
        raw=[auc(labels[i],angle[i],w[i]) for i in range(2)]
        directed=[raw[i] if raw[1-i]>=.5 else 1-raw[i] for i in range(2)]
        results.append(np.array([[raw[i],directed[i],0,0] for i in range(2)]))
        return np.asarray(results)
    point=calc([np.ones(len(s)) for s in splits]);samples,timing=bootstrap(item['id'],binding,lambda rng:calc([cluster_weights(s.subject_id.to_numpy(),rng) for s in splits]))
    weights=np.array([len(s) for s in splits],float);weights/=weights.sum()
    metrics={m:{name:pool_summary(point[j,:,k],samples[:,j,:,k],weights) for k,name in enumerate(('baseline_auc','augmented_auc','delta_auc','residual_conditional_auc'))} for j,m in enumerate(('B0','B1','B2','B3'))}
    for j,m in enumerate(('B0','B1','B2','B3')):
        np.testing.assert_allclose(samples[:,j,:,1]-samples[:,j,:,0],samples[:,j,:,2],atol=1e-14)
    correlations={}
    for population,f in [('available_development',frame.dropna(subset=[marker,ANGLE])),('common_cases',complete)]:
        correlations[population]={}
        for split,s in [('pooled',f),('train',f.loc[f.split=='train']),('validation',f.loc[f.split=='validation'])]:
            rho=float(spearmanr(s[marker],s[ANGLE]).statistic);correlations[population][split]={'n':len(s),'rho':rho if np.isfinite(rho) else None}
    flow={}
    for split,s in complete.groupby('split'):
        o=observed.loc[observed.split==split];previous=o.dropna(subset=[c for c in required if c!=ANGLE])
        direction=r.result_direction(complete,marker,split)
        flow[split]={'N0_n':len(o),'previous_complete_n':len(previous),'complete_n':len(s),'positive':int(s.label.sum()),'negative':int((s.label==0).sum()),'subjects':int(s.subject_id.nunique()),'raw_marker_auc_new':auc(s.label.to_numpy(),direction*s[marker].to_numpy()),'raw_marker_auc_previous':auc(previous.label.to_numpy(),r.result_direction(observed,marker,split)*previous[marker].to_numpy())}
    missing={c:{split:{'missing':int(s[c].isna().sum()),'positive_missing':int(s.loc[s[c].isna(),'label'].sum()),'negative_missing':int((s.loc[s[c].isna(),'label']==0).sum())} for split,s in observed.groupby('split')} for c in required}
    return {'metrics':metrics,'angle_auc_raw':pool_summary(point[4,:,0],samples[:,4,:,0],weights),'angle_auc_directed':pool_summary(point[4,:,1],samples[:,4,:,1],weights),'correlations':correlations,'flow':flow,'missingness':missing,'timing':timing,'B2_equals_B1':comp['B2_equals_B1'],'historical_N2':old['N2']['metrics']}

def report(items,results):
    rows=[];lines=['# AW-034 - Ergebnis B3','',f'Aktualisiert {now()}. Spezifikation SHA-256 `{r.sha(SPEC)}`.', '', 'Fold A train→validation, B validation→train; Pool mit festen Fallzahlgewichten. ΔAUC und bedingte Residual-AUC mit 95%-Perzentil-KI aus 1.000 gemeinsamen Patientenreplikaten und vollständigem Neufit. Neue identische vollständige Fälle B0-B3/Marker. Spearman deskriptiv auf Entwicklungsdaten. Keine unabhängige Prüfung; kein Testzugriff. Keine Wertung über diese vorab festgelegten Größen hinaus.', '', '| Befund | Status | n A/B | B2 ΔAUC [95%-KI] | B3 ΔAUC [95%-KI] | rho | Winkel-AUC |','|---|---|---|---|---|---|---|']
    for item in items:
        ident=item['id'];res=results.get(ident,{});row={'id':ident,'klinikfeld':item['klinikfeld'],**(item['key'] or {}),'status':res.get('status','wartend'),'exception':res.get('exception',''),'selection':'; '.join(item['reasons']),'result_file':str(OUT/ident/(ident+'.json')),'result_sha256':r.sha(OUT/ident/(ident+'.json')) if res else '', 'akte_sha256':item['akte_sha256']}
        data=res.get('data',{})
        for m,metrics in data.get('metrics',{}).items():
            for metric,values in metrics.items():
                for k,v in values.items():row[f'{m}_{metric}_{k}']=json.dumps(v) if isinstance(v,list) else v
        for k in ('angle_auc_raw','angle_auc_directed','correlations','flow','missingness','timing','historical_N2'):row[k]=json.dumps(data.get(k,{}),ensure_ascii=False)
        for trap in ('F1','F2','F3','F4','F5','F6','F7','F8'):row[trap+'_akte']=item['fallen_akte'].get(trap,{}).get('status','offen')
        row['F1_F8_nachrechnung']=json.dumps(res.get('fallen',{}),ensure_ascii=False);rows.append(row)
        def formatted(m):
            v=data.get('metrics',{}).get(m,{}).get('delta_auc',{});return f'{v.get("pooled", "")} {v.get("pooled_ci95", "")}'
        flow=data.get('flow',{});counts='/'.join(str(flow.get(s,{}).get('complete_n','')) for s in ('validation','train'))
        lines.append(f'| {ident} | {row["status"]}: {row["exception"]} | {counts} | {formatted("B2")} | {formatted("B3")} | {data.get("correlations",{}).get("common_cases",{}).get("pooled",{}).get("rho", "")} | {data.get("angle_auc_directed",{}).get("pooled", "")} |')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (r.PROJECT/'nachrechnung/ergebnis_b3.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter=';');w.writeheader();w.writerows(rows)
    lines+=['','CSV: alle B0-B3-Grund-/Markermodell-AUC, ΔAUC, bedingte AUC, A/B/Pool-KI/Kovarianz, historische B0-B2-Basis, Fallfluss/Fehlwerte und Winkel-AUC. Resultat-JSON: Schlüssel, vollständige Hashbindung, Ressourcen, F1-F8. Eingänge: A155-2026-10-06-single-markers, attestierte Entwicklungslabels, AW033-Projektionen, train/validation-Featurepartitionen; Rezept python -m werkzeuge.nachrechnung.b3 --run.', '', 'F1 Aktenposition/Referenz unverändert; F2 selektiertes Kollektiv, keine PPV/NPV; F3 keine Arbeitspunktwahl; F4 Vorselektion/Multiplizität weiter offen; F5 Missingness ausgewiesen, informative Verfügbarkeit offen; F6 zusätzlicher stetiger Winkelkomparator, klinisch stärkster Gesamtvergleich nicht bewiesen; F7 Index unverändert, Verifikationsbias offen; F8 HF/QRS/QTc kontrolliert, technische Robustheit/Ära/Gerät offen. Vertragsausnahmen sind keine Nullbefunde.']
    (r.PROJECT/'nachrechnung/ERGEBNIS_B3.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');args=p.parse_args()
    guard();r.OUT=OUT
    if args.prepare:prepare();return
    inventory=json.loads((OUT/'inventory_AW034.json').read_text(encoding='utf-8'));assert r.sha(SPEC)==inventory['spec_sha256']
    items=inventory['items'];base=r.freeze_binding();base['b3_spec_sha256']=r.sha(SPEC);base['b3_code_sha256']=r.sha(__file__);base['inventory_sha256']=r.sha(OUT/'inventory_AW034.json')
    release=Path(str(data_path("b.release_sources_a155_survival_nullable_20261007_vcg_biomarker")))
    base['angle_definition_sha256']={str(release/n):r.sha(release/n) for n in ('core/vcg_geometry.py','core/biomarkers/catalog_biomarkers.py','mimic/feature_record.py')}
    for x in items:assert r.sha(x['akte'])==x['akte_sha256'],'Akte seit Freeze verändert'
    results={}
    with Budget(OUT) as budget:
        ap=OUT/'angles_development.parquet';mp=OUT/'angles_manifest.json'
        if ap.exists() and mp.exists():
            import pyarrow.parquet as pq
            am=json.loads(mp.read_text(encoding='utf-8'));assert r.sha(ap)==am['projection_file_sha256']
            for part in am['partitions']:
                source=Path(part['path']);assert source.stat().st_size==part['size'] and source.stat().st_mtime_ns==part['mtime_ns']
            a=pq.read_table(ap,columns=['subject_id','study_id','split',ANGLE]).to_pandas()
        else:a=angles(items)
        base['angle_manifest_sha256']=r.sha(mp)
        for item in items:
            guard();ident=item['id'];path=OUT/ident/(ident+'.json');print(json.dumps({'phase':'start','befund':ident}),flush=True)
            if path.exists():
                res=json.loads(path.read_text(encoding='utf-8'));assert res['base']==base
                assert r.sha(path)==json.loads(path.with_suffix('.sha256.json').read_text(encoding='utf-8'))['sha256']
            else:
                res={'auftrag':'AW-034','id':ident,'key':item['key'],'base':base,'started_at':now(),'sealed_read':False}
                if item['pre_status']!='berechenbar':
                    res.update(status=item['pre_status'],exception='')
                    oldpath=OLD/ident/(ident+'.json')
                    if oldpath.exists():res['exception']=json.loads(oldpath.read_text(encoding='utf-8')).get('exception','')
                else:
                    frame,old,oldbinding=verified_old(item)
                    frame=frame.merge(a,on=['subject_id','study_id','split'],how='left',validate='one_to_one')
                    binding={**base,**oldbinding,'joined_projection_hash':r.arrow_sha(__import__('pyarrow').Table.from_pandas(frame,preserve_index=False))}
                    res['binding']=binding;res['data']=calculate(frame,old,item,binding);res['status']='completed'
                    del frame,old;gc.collect()
                res['fallen']={f'F{i}':s for i,s in enumerate(['Aktenposition/Referenz unverändert','Selektierte Krankenhaus-Complete-cases, keine PPV/NPV','Keine Arbeitspunktwahl','Vorselektion/Multiplizität/Auswahloptimismus offen','Fallfluss/Fehlwerte ausgewiesen; informative Missingness offen','B3 zusätzlich Winkel, stärkster klinischer Gesamtvergleich nicht bewiesen','Index unverändert, Verifikationsbias offen','HF/QRS/QTc kontrolliert; technische Artefakte/Gerät/Ära offen'],1)}
                res.update(finished_at=now(),resources=budget.summary());r.write(path,res);r.write(path.with_suffix('.sha256.json'),{'sha256':r.sha(path)})
            results[ident]=res;report(items,results)
        r.write(OUT/'verification_AW034.json',{'finished_at':now(),'result_sha256':{i:r.sha(OUT/i/(i+'.json')) for i in results},'resources':budget.summary(),'base':base,'sealed_read':False})
    print(json.dumps({'phase':'finished','statuses':{s:sum(x['status']==s for x in results.values()) for s in set(x['status'] for x in results.values())}}),flush=True)

if __name__=='__main__':main()
