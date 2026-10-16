"""AW-025: Auswertung erst nach bestandenem Vollgate; keine verkleinerte Testfamilie."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import json, time
from pathlib import Path
import numpy as np
from .aw025 import ROUND, PROJECT, read_small, wait_memory, install_guard, bindings
from .lauf import sha, checkpoint, Budget
from .adapter import ORIGINALS
from .vertrag import ACTIVE
from .vergleich import simultaneous, agreement, icc_sdc

FOLDERS=('VF-01_qrs-t-kopplung','VF-02_zaehlproxy-reparatur')
ORIGINAL_MAP=[ORIGINALS[0] if i!=1 else ORIGINALS[1] for i in range(8)]+[ORIGINALS[3] if i in (1,2,6) else ORIGINALS[2] for i in range(8)]

class WeightedAUC:
    """Fest sortierte Scores, gewichtetes Mann-Whitney mit exakten Bindungen."""
    def __init__(self,y,score):
        self.y=np.asarray(y);self.score=np.asarray(score,float)
        self.keep=np.flatnonzero(np.isfinite(self.score)&np.isfinite(self.y))
        self.order=self.keep[np.argsort(self.score[self.keep],kind='stable')]
        values=self.score[self.order]
        self.starts=np.r_[0,np.flatnonzero(np.diff(values)!=0)+1] if len(values) else np.array([],int)
    def __call__(self,weights):
        if not len(self.order):return np.nan
        w=np.asarray(weights)[self.order];y=self.y[self.order]
        pos=np.add.reduceat(w*(y==1),self.starts);neg=np.add.reduceat(w*(y==0),self.starts)
        denominator=pos.sum()*neg.sum()
        return float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/denominator) if denominator else np.nan

def paired_estimators(y,a,b):
    keep=np.isfinite(a)&np.isfinite(b)
    return WeightedAUC(y,np.where(keep,a,np.nan)),WeightedAUC(y,np.where(keep,b,np.nan))

def polarity(y,x):
    auc=WeightedAUC(y,x)(np.ones(len(y)))
    return -1 if np.isfinite(auc) and auc<.5 else 1

def cluster_draws(y,estimators,replicates=10000):
    """Ein Index-EKG je Person: patientenweiser, endpunktgeschichteter Bootstrap."""
    rng=np.random.default_rng(20261008);groups=[np.flatnonzero(y==v) for v in (0,1)]
    point=np.array([a(np.ones(len(y)))-b(np.ones(len(y))) for a,b in estimators])
    draws=np.empty((replicates,len(estimators)))
    for r in range(replicates):
        weights=np.zeros(len(y))
        for g in groups:weights+=np.bincount(rng.choice(g,len(g),replace=True),minlength=len(y))
        draws[r]=[a(weights)-b(weights) for a,b in estimators]
    return point,draws

def correction_contract(point,draws):
    """Kein Ersatz durch 32/16 Kontraste, falls B2 oder Messpaare fehlen."""
    try:
        result=simultaneous(point,draws,48)
        return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
    except ValueError as exc:return {'status':'nicht entscheidbar','reason':str(exc),'expected_contrasts':48}

def supplement_context(sample):
    """Nur die explizite Pipeline-HF; kein Surrogat aus Gerätereport."""
    import pyarrow.parquet as pq, pyarrow.dataset as ds
    import pandas as pd
    target=ROUND/'vergleich_kontext.parquet';meta=ROUND/'vergleich_kontext.json'
    if target.exists():
        info=json.loads(meta.read_text(encoding='utf-8'))
        if sha(target)!=info['sha256']:raise ValueError('Vergleichskontexthash')
        return read_small(target)
    sources=json.loads((ROUND/'referenzen.json').read_text(encoding='utf-8'))['sources']
    pieces=[];provenance=[]
    for item in sources:
        wait_memory();path=Path(item['path']);schema=pq.ParquetFile(path).schema_arrow
        columns=['study_id']+[c for c in schema.names if c in ('context_heart_rate_bpm','qt_interval_ms','qtc_bazett_ms','qtc_fridericia_ms')]
        table=ds.dataset(str(path),format='parquet').to_table(columns=columns,filter=ds.field('study_id').isin(sample.study_id.tolist()))
        if table.num_rows:
            pieces.append(table.to_pandas());provenance.append({'path':str(path),'sha256':item['sha256'],'columns':columns,'filter':'study_id IN AW023b frozen sample','rows':table.num_rows})
    frame=pd.concat(pieces,ignore_index=True)
    if frame.study_id.duplicated().any() or len(frame)!=len(sample):raise ValueError('Vergleichskontextidentität')
    import pyarrow as pa
    pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),target,compression='zstd')
    checkpoint(meta,{'sha256':sha(target),'sources':provenance})
    return frame

def analytical_contract():
    """Vorab spezifizierte Kurven, kein klinisches Signal und keine Parameterwahl."""
    from werkzeuge.tests.test_varianten import curve
    from .metriken import repaired_qrs,graph_statistics
    from .vergleich import invariance
    from scipy.spatial.distance import cdist
    records=[]
    for kind in ('circle','ellipse','hairpin','branches'):
        reference=repaired_qrs(curve(8192,kind))
        for n in (32,64,128,256,512):
            for irregular in (False,True):
                for scale in (.5,1,2):
                    x=curve(n,kind,irregular)*scale;value=repaired_qrs(x)
                    d=cdist(x,x);i,j=np.triu_indices(n,5);dist=d[i,j]
                    graph=graph_statistics((d>0)&(d<np.percentile(dist,10)))
                    passes={k:invariance(reference[k],value[k],fraction=k in ('near','graph_fixed')) for k in ('near','graph_p10','graph_fixed','events')}
                    records.append({'curve':kind,'n':n,'nonuniform':irregular,'scale':scale,'repaired':value,'reference':reference,'passes':passes,'original_p5_count':int((dist<=np.percentile(dist,5)).sum()),'original_graph_plus1':max(0,graph['edges']-n+1),'time_scales':[.5,1,2]})
    report={'cases':records,'case_count':len(records),'failed_cases':sum(not all(r['passes'].values()) for r in records),'time_scale_interpretation':'Identische Polygongeometrie, reine Änderung der Zeitmetadaten; keine reale klinische Dauerkontrolle','source_sha256':sha(PROJECT/'werkzeuge/tests/test_varianten.py')}
    checkpoint(ROUND/'messvertrag.json',report)
    return report

def gate_report(state):
    """Keine biologische Schlussfolgerung aus unvollständiger/Gate-gestoppter Rechnung."""
    for family,folder in enumerate(FOLDERS):
        text=f'# AW-025 - {folder}: Runde 1\n\n'
        text+='**Nach Ergebnissicht optimiert, höchstens E1.** Technischer Gate-Stopp; keine klinische oder methodische Verbesserung nachgewiesen.\n\n'
        text+=f"Vollrechnungstatus `{state['status']}`: {state['n_completed']} von 5.648 EKGs als vollständige Blöcke committed. Fehler: `{state.get('error','unbekannt')}`. Die Originaläquivalenz ist ein zwingendes Tor; keine Toleranzänderung und keine Fortsetzung über den Fehler.\n\n"
        text+='| Variante | Original | ΔAUC / KI | Korrigiertes Ergebnis | Messgüte | Fazit und Begründung |\n|---|---|---|---|---|---|\n'
        for i in range(family*8,(family+1)*8):
            text+=f'| {ACTIVE[i]} | {ORIGINAL_MAP[i]} | nicht geprüft | nicht entscheidbar (48/64-Kontrastfamilie nicht schätzbar) | nicht geprüft | nicht entscheidbar: Vollgate nicht bestanden |\n'
        text+='\nDiese Nichtschätzbarkeit ist kein Nullbefund und keine Aussage gleicher AUC. Residualfits, Bootstrap und biologische Auswertung wurden nach dem technischen Stopp nicht auf einer nachträglich verkleinerten Population gerechnet. Wiederholungspaare werden nicht nachgeladen: die eingefrorene Liste enthält genau ein EKG je Person. ICC/SDC sind darauf nicht prüfbar.\n\n'
        text+='Stichprobe: AW023b/stichprobe/liste.parquet; Kontext: AW023c/stichprobe_kontext.parquet; Herkunft aller Fortschritts-/Ressourcenzahlen: <local path>, Rezept werkzeuge/varianten/aw025.py. Fehlerbeleg ausschließlich unter <local path>. Keine Patientenzeilen in diesem Bericht.\n\n'
        text+='F1 offen (Referenzzeit); F2 eingeschränkt (angereicherte Entwicklungsliste, keine PPV/NPV); F3 eingeschränkt (keine Arbeitspunktsuche); F4 eingeschränkt (alle aktiven Varianten aufgeführt, keine Nachauswahl); F5 offen (Messbarkeit nicht vollständig geprüft); F6 offen (N-2/Komparatorvergleich nicht durchgeführt); F7 eingeschränkt (keine Wiederholungs- oder unabhängige Validierung); F8 offen (Originalgate gescheitert). P01a: Stichproben-/Kontext-/Planbindung bestanden; P01b/P01c: Volläquivalenz nicht bestanden. P08: keine inferentiellen Intervalle.\n\n'
        text+='Widerlegung/Stopp: Jede Status-/Missingness-/Metadaten- oder numerische Abweichung verhindert die Freigabe. Offene Nachrechnung: Fehlerklärung durch K, anschließend gesondert gebundene Wiederaufnahme; kein automatischer Retry des Gatefehlers. Sicherheit hoch für den dokumentierten technischen Stopp; keine Sicherheit einer biologischen Verbesserung.\n'
        (PROJECT/'verbesserung'/folder/'ERGEBNIS_RUNDE1.md').write_text(text,encoding='utf-8')

def submission():
    state=json.loads((ROUND/'checkpoint.json').read_text(encoding='utf-8'))
    comparison=ROUND/'vergleich.json'
    text='# AW-025 - Abgabe PR/MS\n\nStand: 09.10.2026; Rechenbeginn 08.10.2026. Auftragserledigung und Fachstatus werden getrennt geführt.\n\n'
    passed=state['status']=='passed'
    text+=('Vollrechnung technisch bestanden; die vorab festgelegten inferentiellen Erfolgsfamilien bleiben wegen fehlender B2-Spezifikation/Wiederholungspaare nicht schätzbar. Kein Verbesserungsclaim.\n\n' if passed else '**Nicht vollständig erledigt: zwingender Äquivalenz-Gate-Stopp.** Keine Fortsetzung über den Fehler, keine biologische Auswertung auf nachträglich verkleinerter Liste, keine Erfolgsaussage.\n\n')
    text+=f"Vollrechnung: Status `{state['status']}`, {state['n_completed']} von 5.648 EKGs in {len(state['completed_blocks'])} committed Blöcken; {state['n_equivalent']} EKGs als abgeschlossene Blöcke äquivalent. Dauer der Runner-Sessions {state['seconds']:.3f} s; gemessene Prozessspitze {state['peak_rss_bytes']} Bytes ({state['peak_rss_bytes']/1024**2:.3f} MiB). Quelle: `<local path>`, Schlüssel n_completed, completed_blocks, n_equivalent, seconds, peak_rss_bytes; Rezept `werkzeuge/varianten/aw025.py`. Nichtcommittete Gatearbeit ist in der Sessionlaufzeit enthalten, aber nicht als vollständiger Block gezählt.\n\n"
    if not passed:text+=f"Gatefehler: `{state.get('error_type')}: {state.get('error')}`. Diagnose/Rechenartefakte ausschließlich unter E:, keine Patientenzeilen in dieser Abgabe.\n\n"
    if passed and comparison.exists():
        info=json.loads(comparison.read_text(encoding='utf-8'))
        text+=f"Alle 16 aktiven Varianten berichtet; train-fixe Residualfits in residualfits.json, 10.000 gemeinsame Patientenbootstrap-Ziehungen in vergleich.json (Seed 20261008). Auswertungsdauer {info['seconds']:.3f} s. Messvertrag: {info['analytical_case_count']} analytische Kontrollen, davon {info['analytical_failed_cases']} nicht bestanden; Herkunft messvertrag.json, Rezept aw025_auswerten. ICC/SDC auf der eingefrorenen Ein-Index-EKG-Liste nicht prüfbar.\n\n"
    text+='Geänderte/neue Projektdateien: `werkzeuge/varianten/aw025.py`, `werkzeuge/varianten/aw025_auswerten.py`, `werkzeuge/tests/test_varianten.py`, beide `verbesserung/VF-0x_…/ERGEBNIS_RUNDE1.md`, diese Abgabe. Datenartefakte ausschließlich `<local path>`. Keine Änderung an Stichprobe, Kontext, Ideenakten, Auftragsbuch, gemeinsamer Pipeline, Befundakten oder gesperrten Daten. Keine Installation/Außenkommunikation.\n\n'
    text+='Reproduktion aus dem Projektordner mit Auswertungs-Python und -<local path>```powershell\n& <local path> -B -m werkzeuge.varianten.aw025 run\n& <local path> -B -m werkzeuge.varianten.aw025_auswerten\n& <local path> -B -m pytest werkzeuge\\tests\\test_varianten.py -q -p no:cacheprovider\n```\n\n'
    text+='Der Runner wartet bei RAM/Commit unter 4 GiB in 60-s-Schritten; Guards nur für diesen Auftrag ersetzt, historischer Rauchtestvertrag unverändert. Ein Python-Prozess, Threadbegrenzung und 1,5-GiB-Prozesswächter. Blockcommit erst nach vollständigem Worker-/Prepare-Gate und Variantenausgabe; Wiederaufnahme prüft Source-/Plan-/Stichproben-/Kontext-/Outputhashes. Nach status=stop_gate ist automatisches Fortsetzen explizit gesperrt.\n\n'
    text+='Offene Fragen an K: technische Gateklärung vor Wiederaufnahme; B2-Modell/Regularisierung ausführbar fixieren; unabhängige Referenzzeitattestation; gesondert eingefrorene Wiederholungsstichprobe. B2- und Messgütekontraste nicht durch kleinere Familien ersetzen. Kein Gleichheits-/Nullclaim aus Nichtschätzbarkeit.\n\n'
    text+='Nebenbefunde: Der bisherige Rauchtest belegt Äquivalenz nur seiner 20 IDs; das Vollgate bleibt eigenständig erforderlich. Die Vergleichsliste enthält ein Index-EKG je Person und kann den geplanten Wiederholungszweig nicht erfüllen. Primärendpunkt reduced_lvef_strict, Entwicklung und interne Bestätigung getrennt; Testdaten nie geöffnet.\n\n'
    text+='F1 offen (Referenzzeit/Definitionskopplung); F2 eingeschränkt (angereichert); F3 eingeschränkt (train-fixe Wahl, keine Schwellenoptimierung); F4 offen (kein korrigierter Erfolg); F5 offen/eingeschränkt (Messbarkeit/Abstention); F6 offen (B2); F7 eingeschränkt (keine Wiederholungen/extern); F8 offen (technisches Gate und Artefaktprüfung). Details in beiden Familienberichten.\n\n'
    text+='Selbsteinschätzung: hohe Sicherheit der dokumentierten Bindung und Stopplogik; keine Grundlage für einen klinischen Verbesserungsclaim. Benötigte zusätzliche Pakete: keine. Abnahme/weiteres Vorgehen durch K erforderlich.\n\n'
    text+='Artefaktbindung:\n\n| Datei | SHA256 |\n|---|---|\n'
    for p in [ROUND/'checkpoint.json',PROJECT/'werkzeuge/varianten/aw025.py',PROJECT/'werkzeuge/varianten/aw025_auswerten.py',PROJECT/'werkzeuge/tests/test_varianten.py']:
        text+=f'| {p.name} | `{sha(p)}` |\n'
    (PROJECT/'koordination/abgaben/AW-025.md').write_text(text,encoding='utf-8')

def evaluate():
    install_guard();wait_memory();started=time.perf_counter()
    state=json.loads((ROUND/'checkpoint.json').read_text(encoding='utf-8'))
    if state['status'] not in ('passed','stop_gate'):
        raise RuntimeError('Vollrechnung läuft/unvollständig; noch keine Ergebnisabgabe')
    if state['status']!='passed':
        gate_report(state);return
    if bindings()!=state['binding']:raise ValueError('Quell-/Plan-/Kontextbindung nach Vollrechnung verändert')
    for block in state['completed_blocks']:
        for key in ('original','variant'):
            if sha(Path(block[key+'_path']))!=block[key+'_sha256']:raise ValueError('Outputblockhash verändert')
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from .residuen import Residualizer
    sample=read_small(Path(str(data_path("b.varianten_aw023b_stichprobe_liste_parquet"))))
    context=read_small(Path(str(data_path("b.varianten_aw023c_stichprobe_kontext_parquet"))))
    ref=read_small(ROUND/'referenzen.parquet')
    extra=supplement_context(sample)
    for c in extra.columns:
        if c!='study_id' and c in ref:extra=extra.drop(columns=c)
    ref=ref.merge(extra,on='study_id',validate='one_to_one')
    analytical=analytical_contract()
    values=pd.concat([read_small(Path(b['variant_path'])) for b in state['completed_blocks']],ignore_index=True)
    originals=pd.concat([read_small(Path(b['original_path'])) for b in state['completed_blocks']],ignore_index=True)
    wide=values.pivot(index='study_id',columns='variant',values='value')
    base=sample.merge(context[['study_id','age','sex']],on='study_id',validate='one_to_one').merge(ref,on=['subject_id','study_id','split'],validate='one_to_one').join(wide,on='study_id')
    if base.subject_id.duplicated().any():raise ValueError('Patientencluster nicht eindeutig')
    # Eindeutige Kontextprojektion, keine Ersatz-HF aus Gerätetexten.
    hf=next((c for c in ('context_heart_rate_bpm','heart_rate_bpm') if c in base),None)
    fit_records=[]
    for variant,target in ((ACTIVE[7],ORIGINALS[0]),(ACTIVE[15],ACTIVE[8])):
        if hf is None or 'qt_interval_ms' not in base:
            fit_records.append({'variant':variant,'status':'nicht schätzbar','reason':'Explizite HF/QT-Projektion fehlt'});continue
        numeric=['qrs_duration_ms','qt_interval_ms',hf,'age']
        complete=np.isfinite(base[numeric].to_numpy(float)).all(1)&base.sex.notna()&np.isfinite(base[target].to_numpy(float))
        train=np.flatnonzero(complete&(base.split=='train'));held=np.flatnonzero(complete&(base.split=='validation'))
        output=np.full(len(base),np.nan);x=base[numeric].to_numpy(float);sex=base.sex.to_numpy();y=base[target].to_numpy(float)
        record={'variant':variant,'target':target,'numeric':numeric,'train_n':len(train),'validation_n':len(held),'folds':[]}
        try:
            fit=Residualizer().fit(x[train],sex[train],y[train]);output[held]=fit.residual(x[held],sex[held],y[held])
            record.update(mean=fit.mean.tolist(),sd=fit.sd.tolist(),categories=fit.categories,coef=fit.model.coef_.tolist(),intercept=float(fit.model.intercept_))
            for tr,ho in GroupKFold(5).split(x[train],groups=base.subject_id.to_numpy()[train]):
                model=Residualizer().fit(x[train[tr]],sex[train[tr]],y[train[tr]])
                output[train[ho]]=model.residual(x[train[ho]],sex[train[ho]],y[train[ho]])
                record['folds'].append({'train_n':len(tr),'held_n':len(ho),'mean':model.mean.tolist(),'sd':model.sd.tolist(),'coef':model.model.coef_.tolist()})
            record['status']='passed'
        except Exception as exc:record.update(status='Abstention',reason=str(exc));output[:]=np.nan
        base[variant]=output;fit_records.append(record)
    checkpoint(ROUND/'residualfits.json',{'fits':fit_records,'sample_sha256':state['binding']['sample_sha256']})
    import pyarrow as pa, pyarrow.parquet as pq
    final_values=values.copy()
    for v in (ACTIVE[7],ACTIVE[15]):
        mask=final_values.variant==v
        final_values.loc[mask,'value']=final_values.loc[mask,'study_id'].map(base.set_index('study_id')[v])
        final_values.loc[mask,'abstention']=np.where(final_values.loc[mask,'value'].notna(),'','Residuum nicht schätzbar: fehlende Kovariaten/Original oder Fit-Abstention; siehe residualfits.json')
    final_path=ROUND/'varianten_final.parquet'
    pq.write_table(pa.Table.from_pandas(final_values,preserve_index=False),final_path,compression='zstd')
    checkpoint(ROUND/'varianten_final.json',{'path':str(final_path),'sha256':sha(final_path),'rows':len(final_values),'variants':list(ACTIVE),'residual_fit_sha256':sha(ROUND/'residualfits.json')})
    # Keine vorgegebene B2-Modellfamilie/Regularisierung im eingefrorenen Plan.
    # Fehlender ausführbarer Vertrag darf nicht durch einen eigenen Fit ersetzt werden.
    clinical_reason='B2-Schätzmodell/Regularisierung im Vergleichsplan nicht ausführbar festgelegt; kein stiller Ersatzfit'
    train=base.split=='train';validation=base.split=='validation'
    ytrain=base.loc[train,'label'].to_numpy();yv=base.loc[validation,'label'].to_numpy()
    candidates=[c for c in ('qrs_duration_ms','frontal_xy_qrs_t_angle_deg','spatial_qrs_t_angle_deg') if c in base]
    candidate_aucs={}
    for c in candidates:
        x=base.loc[train,c].to_numpy(float);scores=np.full(len(x),np.nan)
        for tr,ho in GroupKFold(5).split(x,groups=base.loc[train,'subject_id']):scores[ho]=polarity(ytrain[tr],x[tr])*x[ho]
        candidate_aucs[c]=WeightedAUC(ytrain,scores)(np.ones(len(x)))
    comparator=sorted(candidates,key=lambda c:(-candidate_aucs[c] if np.isfinite(candidate_aucs[c]) else np.inf,c))[0]
    checkpoint(ROUND/'vergleich_manifest.json',{'candidate_train_oof_auc':candidate_aucs,'selected_comparator':comparator,'clinical_family':48,'measurement_family':64,'alpha_each':.025,'replicates':10000,'b2_status':clinical_reason,'repeat_pairs':0,'plan_hashes':state['binding']['plans']})
    # Alle 48 Plätze bleiben erhalten. B2 ist explizit NaN, keine Verkleinerung.
    estimators=[];results=[];blank=WeightedAUC(yv,np.full(len(yv),np.nan))
    cp=polarity(ytrain,base.loc[train,comparator]);compv=cp*base.loc[validation,comparator].to_numpy(float)
    all_complete=np.isfinite(base[list(ACTIVE)].to_numpy(float)).all(1)
    for i,v in enumerate(ACTIVE):
        original=ORIGINAL_MAP[i];xt=base.loc[train,v].to_numpy(float);ot=base.loc[train,original].to_numpy(float)
        pv=polarity(ytrain,xt);po=polarity(ytrain,ot)
        a=pv*base.loc[validation,v].to_numpy(float);b=po*base.loc[validation,original].to_numpy(float)
        ap,bp=paired_estimators(yv,a,b);ac,bc=paired_estimators(yv,a,compv)
        common=all_complete[validation.to_numpy()]
        aa,bb=paired_estimators(yv,np.where(common,a,np.nan),np.where(common,b,np.nan))
        estimators.extend([(ap,bp),(ac,bc),(blank,blank)])
        pair=np.isfinite(a)&np.isfinite(b)
        # OOF-Polung für die reine train-interne Richtungsprüfung.
        at=np.full(len(xt),np.nan);bt=at.copy()
        for tr,ho in GroupKFold(5).split(xt,groups=base.loc[train,'subject_id']):
            train_values=xt[tr]
            # Residualpolung ebenfalls ausschließlich aus dem aktuellen Fitfold.
            # Bereits OOF-residualisierte Trainingswerte können Modelle enthalten,
            # die den aktuellen held-out-Fold sahen; diese hier nicht wiederverwenden.
            if v in (ACTIVE[7],ACTIVE[15]) and np.isfinite(xt).any() and hf is not None:
                target=ORIGINALS[0] if v==ACTIVE[7] else ACTIVE[8]
                tx=base.loc[train,['qrs_duration_ms','qt_interval_ms',hf,'age']].to_numpy(float)
                ts=base.loc[train,'sex'].to_numpy();ty=base.loc[train,target].to_numpy(float)
                good=np.isfinite(tx[tr]).all(1)&np.isfinite(ty[tr])&pd.notna(ts[tr])
                train_values=np.full(len(tr),np.nan)
                try:
                    m=Residualizer().fit(tx[tr][good],ts[tr][good],ty[tr][good])
                    train_values[good]=m.residual(tx[tr][good],ts[tr][good],ty[tr][good])
                except Exception:pass
            at[ho]=polarity(ytrain[tr],train_values)*xt[ho];bt[ho]=polarity(ytrain[tr],ot[tr])*ot[ho]
        ta,tb=paired_estimators(ytrain,at,bt)
        results.append({'variant':v,'original':original,'polarity_variant':pv,'polarity_original':po,
            'validation_pairs':int(pair.sum()),'validation_positive':int((pair&(yv==1)).sum()),'validation_negative':int((pair&(yv==0)).sum()),
            'variant_auc_paired':ap(np.ones(len(yv))),'original_auc_paired':bp(np.ones(len(yv))),
            'variant_auc_available':WeightedAUC(yv,a)(np.ones(len(yv))),
            'original_auc_available':WeightedAUC(yv,b)(np.ones(len(yv))),
            'train_variant_available':int(np.isfinite(xt).sum()),'validation_variant_available':int(np.isfinite(a).sum()),
            'train_oof_delta':ta(np.ones(len(ytrain)))-tb(np.ones(len(ytrain))),
            'validation_all_variant_common_n':int(common.sum()),
            'validation_all_variant_common_positive':int((common&(yv==1)).sum()),
            'validation_all_variant_common_delta':aa(np.ones(len(yv)))-bb(np.ones(len(yv)))})
    point,draws=cluster_draws(yv,estimators)
    np.save(ROUND/'auc_bootstrap.npy',draws,allow_pickle=False)
    corrected=correction_contract(point,draws)
    for i,r in enumerate(results):
        d=draws[:,3*i];finite=d[np.isfinite(d)]
        r['delta_auc']=float(point[3*i]);r['uncorrected_ci_95']=np.quantile(finite,[.025,.975]).tolist() if len(finite)>=9500 else None
        r['delta_comparator']=float(point[3*i+1]);r['fold_consistent']=bool(r['train_oof_delta']*r['delta_auc']>0)
        r['decision']='nicht entscheidbar';r['measurement']='ICC/SDC nicht prüfbar: ein EKG je Person'
    transforms=[]
    for name,a,b in (('Frechet Dower/Kors',ACTIVE[0],ORIGINALS[0]),('Graph64Fix Dower/Kors',ACTIVE[14],ACTIVE[10])):
        for fold in ('train','validation'):
            part=base[base.split==fold];x=part[b].to_numpy(float);z=part[a].to_numpy(float);keep=np.isfinite(x)&np.isfinite(z)
            from scipy.stats import spearmanr
            uncertainty=None;difference=z[keep]-x[keep]
            if len(difference)>=3:
                rng=np.random.default_rng(20261008);samples=np.empty((10000,3))
                for r in range(10000):
                    d=difference[rng.integers(0,len(difference),len(difference))]
                    bias=d.mean();sd=d.std(ddof=1);samples[r]=[bias,bias-1.96*sd,bias+1.96*sd]
                uncertainty={k:np.quantile(samples[:,j],[.025,.975]).tolist() for j,k in enumerate(('bias','loa_lower','loa_upper'))}
            train_sd=float(base.loc[train,b].std(ddof=1))
            transforms.append({'contrast':name,'fold':fold,'pairs':int(keep.sum()),'spearman':float(spearmanr(x[keep],z[keep]).statistic),'agreement':agreement(x[keep],z[keep]) if keep.sum()>=3 else None,'bootstrap_ci_95':uncertainty,'train_original_sd':train_sd,'bias_train_sd_units':float(difference.mean()/train_sd) if len(difference) and train_sd>0 else None,'inference':'Patientenbootstrap, 10000 Ziehungen, unadjustierte deskriptive KIs; LoA selbst kein KI, keine klinische Austauschbarkeit'})
    result={'clinical_correction':corrected,'b2_reason':clinical_reason,'measurement_correction':{'status':'nicht prüfbar','expected_contrasts':64,'repeat_pairs':0},'results':results,'transformations':transforms,'analytical_case_count':analytical['case_count'],'analytical_failed_cases':analytical['failed_cases'],'seconds':time.perf_counter()-started}
    checkpoint(ROUND/'vergleich.json',result)
    for family,folder in enumerate(FOLDERS):
        text=f'# AW-025 - {folder}: Runde 1\n\n**Nach Ergebnissicht optimiert, höchstens E1.** Alle acht aktiven Varianten; keine Nachauswahl.\n\n'
        text+='Validation = interne Bestätigung; train = Entwicklung mit patientenweiser OOF-Richtungsprüfung. Keine unabhängige Prüfung. Klinische Referenzzeit bleibt unbestätigt.\n\n'
        text+='| Variante | gepaarte n (+/−) | AUC Original / Variante | ΔAUC (unkorrigiertes 95%-KI) | train-OOF Δ / gleiche Richtung | Korrektur / Messgüte | Fazit |\n|---|---|---|---|---|---|---|\n'
        for r in results[family*8:(family+1)*8]:
            def f(x):return f'{x:.6f}' if np.isfinite(x) else 'nicht schätzbar'
            ci=r['uncorrected_ci_95'];interval='nicht schätzbar' if ci is None else f'[{f(ci[0])}; {f(ci[1])}]'
            text+=f"| {r['variant']} | {r['validation_pairs']} ({r['validation_positive']}/{r['validation_negative']}) | {f(r['original_auc_paired'])} / {f(r['variant_auc_paired'])} | {f(r['delta_auc'])} {interval} | {f(r['train_oof_delta'])} / {r['fold_consistent']} | 48/64-Familie nicht schätzbar; ICC/SDC nicht prüfbar | nicht entscheidbar |\n"
        text+=f'\nBegründung aller Fazits: {clinical_reason}. Kein Austausch der vorab festgelegten max-|T|-Familie gegen eine kleinere Familie. Unkorrigierte Intervalle sind deskriptiv und erlauben keinen Gewinnerclaim. Messgütefamilie mit 64 Kontrasten ebenfalls nicht schätzbar, weil die Stichprobe keine Wiederholungspaare enthält.\n\n'
        text+=f'Stärkster stetiger Komparator aus train-OOF: `{comparator}`; Kandidaten und AUCs in vergleich_manifest.json. Gepaarte Komparatordifferenzen, Polung, Verfügbarkeit, Residualfitkoeffizienten und Transformationsvergleich in <local path> und residualfits.json. Alle Zahlen stammen aus diesen Artefakten, Filter variant und fold=validation bzw. train; Rezept werkzeuge/varianten/aw025_auswerten.py. 10.000 gemeinsame endpunktgeschichtete Patientenbootstrap-Ziehungen, Seed 20261008; ein Index-EKG je Person.\n\n'
        text+='Transformationsrobustheit: explizite Kors/Dower-Kontraste für Fréchet und reparierte Graphdichte, deskriptiv; native Achtkanalspiegeldistanz besitzt einen anderen Messraum. Analytische Abtast-/Längeninvarianz ist getrennt anhand AW023_SYNTHETIK.json dokumentiert; diese technischen Kontrollen belegen keinen klinisch dauerunabhängigen Rest.\n\n'
        text+='F1 offen (Referenzzeit/Dauerkopplung); F2 eingeschränkt (angereichert, keine PPV/NPV); F3 eingeschränkt (train-fixe Polung, keine Schwelle); F4 offen (48/64-Korrektur nicht schätzbar, kein Erfolg); F5 eingeschränkt (paarweise Nenner/Abstention, All-Varianten-Sensitivität); F6 offen (B2 nicht ausführbar festgelegt); F7 eingeschränkt (ein Index-EKG, keine Wiederholung/extern); F8 offen (Messvertragsprüfung und realer Dauerrest getrennt). P01a/b/c: volle technische Äquivalenz laut checkpoint; P08: nur unkorrigierte gepaarte Patientenintervalle, kein simultanes KI.\n\n'
        text+='Offene Frage an K: ausführbares B2-Schätzmodell mit vorab fixierter Regularisierung und eigenem Folgeauftrag; unabhängige Referenzzeitattestation; separate Wiederholungsstichprobe. Widerlegung: fehlender bereinigter Rest oder korrigierte Untergrenze ohne Erfolg schließt Verbesserungsclaim aus. Sicherheit hoch für technische Reproduktion, mittel für deskriptiven Entwicklungsvergleich, keine klinische Freigabe.\n'
        (PROJECT/'verbesserung'/folder/'ERGEBNIS_RUNDE1.md').write_text(text,encoding='utf-8')

if __name__=='__main__':
    install_guard();wait_memory()
    with Budget() as budget:evaluate();submission()
