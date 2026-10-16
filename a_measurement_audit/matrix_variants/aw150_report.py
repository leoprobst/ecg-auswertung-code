"""AW150: aggregierte Auswertung, transparente Schätzbarkeit und deutsche Abgabe."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json, hashlib
from werkzeuge.varianten.aw150_analysis import OUT,PROJECT,VAR,CAT,S2,RC,CLASSES,ATTRS,readj,sha,save,readp,combine,counts
import numpy as np,pandas as pd
from werkzeuge.transform_katalog.metriken import vergleichen

FIG=PROJECT/'publikationen/PUB-01_messvaliditaet/abbildungen/v11'

def icc(x,y):
    ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok];n=len(x)
    if n<30:return np.nan
    scale=max(float(np.std(np.r_[x,y])),np.finfo(float).tiny);center=float(np.mean(np.r_[x,y]))
    x=(x-center)/scale;y=(y-center)/scale
    mean=(x+y)/2;grand=mean.mean()
    msr=2*np.sum((mean-grand)**2)/(n-1)
    msc=n*((x.mean()-grand)**2+(y.mean()-grand)**2)
    mse=(np.sum((x-mean-x.mean()+grand)**2)+np.sum((y-mean-y.mean()+grand)**2))/(n-1)
    den=msr+mse+2*(msc-mse)/n
    return float((msr-mse)/den) if den>0 else np.nan

def textnum(x,digits=4):
    return 'nicht schätzbar' if pd.isna(x) else f'{x:.{digits}f}'

def table(frame,cols):
    g=frame[cols].copy().fillna('')
    return '| '+' | '.join(cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'+'\n'.join('| '+' | '.join(str(v).replace('|','\\|') for v in r)+' |' for r in g.itertuples(index=False,name=None))

def sampling_qc(manifest):
    """Apply the unchanged loader's raw-signal QC also at 250 Hz.

    Marker calculation does not consume the WaveformLoadResult quality object;
    gate the stored 250-Hz values here with the original release eligibility.
    Each case is checkpointed. No additional signal or person data in project.
    """
    from werkzeuge.varianten.adapter import release_import
    from werkzeuge.varianten.aw150_analysis import gate
    import pyarrow.parquet as pq
    from scipy.signal import resample_poly
    fr=release_import()
    from vcg_biomarker.mimic.signals import repair_short_gaps,assess_signal_quality
    cols=pq.ParquetFile(OUT/'selection_private.parquet').schema_arrow.names
    selection=readp(OUT/'selection_private.parquet',cols)
    eligible=[];mhash=sha(OUT/'measurement_manifest.json')
    for i,row in enumerate(selection.to_dict('records')):
        p=OUT/'sampling_qc'/f'{i:04d}.json'
        if p.exists():
            r=readj(p);assert r['manifest_sha256']==mhash
        else:
            gate();loaded=fr.MimicWaveformLoader(target_sampling_rate=500).load(row['waveform_path'],expected_subject_id=int(row['subject_id']),expected_study_id=int(row['study_id']))
            raw=resample_poly(loaded.signal_millivolts,1,2,axis=0,window=np.array(manifest['filter_coefficients']),padtype='constant')
            raw,repair=repair_short_gaps(raw,250);quality=assess_signal_quality(raw,250)
            r={'manifest_sha256':mhash,'fs250_usable':bool(quality.usable and not quality.limb_lead_reversal_suspected),'score':quality.score,'repaired_samples':int(repair.repaired_samples),'original500':bool(loaded.original_sampling_rate==500)}
            save(str(p.relative_to(OUT)),r)
        eligible.append(r['fs250_usable'] and r['original500'])
    save('sampling_qc_summary.json',{'n':500,'eligible':int(sum(eligible)),'rule':'unchanged Release repair_short_gaps and assess_signal_quality; usable and no suspected limb reversal; original500 only','manifest_sha256':mhash})
    return np.array(eligible,bool)

def report():
    manifest=readj(OUT/'measurement_manifest.json');mhash=sha(OUT/'measurement_manifest.json')
    files=sorted((OUT/'measurements').glob('*.json'));assert len(files)==500
    records=[readj(p) for p in files];assert all(r['manifest_sha256']==mhash for r in records)
    assert [r['row_index'] for r in records]==list(range(500))
    names=manifest['full_marker_ids'];panel=manifest['panel_marker_ids'];cat=pd.read_csv(CAT)
    baseline=np.full((500,len(names)),np.nan);s4=np.full((6,500,len(names)),np.nan)
    ss=np.full((6,500,len(panel)),np.nan);rate=np.full((500,len(panel)),np.nan);fixed=np.full_like(rate,np.nan)
    folds=np.array([r['split'] for r in records]);computed=np.zeros(500,bool);original500=np.zeros(500,bool)
    seconds=0
    for i,r in enumerate(records):
        if r['status']!='computed':continue
        assert r['baseline_equivalent'];computed[i]=True;original500[i]=r['source_500_hz'];seconds+=r['seconds']
        baseline[i]=r['baseline']
        assert len(r['NS05'])==len(r['S'])==6
        for j in range(6):
            assert (r['NS05'][j]['boundary'],r['NS05'][j]['sign'])==(ATTRS[j//2],(-1,1)[j%2])
            s4[j,i]=r['NS05'][j]['values'];ss[j,i]=r['S'][j]['values']
        if original500[i]:rate[i]=r['fs250'];fixed[i]=r['fs250_fixed']
    qc250=sampling_qc(manifest)
    rate[~qc250]=np.nan;fixed[~qc250]=np.nan
    NS05=[];S=[];sampling=[];rng=np.random.default_rng(manifest['seed']);keyids={'qrs_t_normalized_discrete_frechet','qrs_t_frechet_beat_median','qrs_t_normalized_dtw_distance','spatial_qrs_t_angle_deg','qrs_duration_ms','qtc_fridericia_ms'}
    for k,n in enumerate(names):
        for j in range(6):
            for f in ('train','validation','pool'):
                take=np.ones(500,bool) if f=='pool' else folds==f
                m=vergleichen(baseline[take,k],s4[j,take,k])
                NS05.append({'marker_id':n,'boundary':ATTRS[j//2],'sign':(-1,1)[j%2],'fold':f,**m})
    ns=pd.DataFrame(NS05);ns.to_csv(OUT/'ns05_boundary4_metrics.csv',index=False)
    nsrows=[]
    for n,g in ns.groupby('marker_id',sort=False):
        r={'marker_id':n};overall=[]
        for j in range(6):
            z=g.loc[g.boundary.eq(ATTRS[j//2])&g.sign.eq((-1,1)[j%2])].set_index('fold')
            c=combine(z.loc['train','class'],z.loc['validation','class']);overall.append(c)
            prefix='boundary4_'+ATTRS[j//2]+('_minus' if j%2==0 else '_plus')
            r[prefix+'_class']=c
            for f in ('train','validation','pool'):
                for key in ('class','n_pairs','loa_low','loa_high','bias','spearman','relative_loa_width','quintile_switch'):
                    r[prefix+'_'+f+'_'+key]=z.loc[f].get(key)
        r['boundary4_worst_class']=max(overall,key=CLASSES.get) if all(c in CLASSES for c in overall) else ('konstant_identisch' if set(overall)=={'konstant_identisch'} else 'nicht_schaetzbar')
        r['boundary4_stable_directions']=overall.count('stabil')
        nsrows.append(r)
    nswide=pd.DataFrame(nsrows)
    base=pd.read_csv(OUT/'matrix_base_aggregate.csv')
    for j in range(6):
        for k,n in enumerate(panel):
            col=names.index(n);x=baseline[:,col];y=ss[j,:,k];ok=np.isfinite(x)&np.isfinite(y);delta=y[ok]-x[ok]
            rcg=base.loc[base.marker_id.eq(n),'repeat_gesamt_rc'].iloc[0];rcp=base.loc[base.marker_id.eq(n),'repeat_proxy_rc'].iloc[0]
            m=vergleichen(x,y)
            r={'marker_id':n,'boundary':ATTRS[j//2],'sign':(-1,1)[j%2],'actual_ms':(-1,1)[j%2]*(14,12,32)[j//2],**m,'mean_absolute_change':float(np.abs(delta).mean()) if len(delta) else np.nan,'p95_absolute_change':float(np.quantile(np.abs(delta),.95)) if len(delta) else np.nan,'repeat_gesamt_rc':rcg,'repeat_proxy_rc':rcp}
            for pop,rc in (('gesamt',rcg),('proxy',rcp)):
                r['mac_over_rc_'+pop]=r['mean_absolute_change']/rc if rc>0 else np.nan
                r['halfwidth_over_rc_'+pop]=1.96*m.get('difference_sd',np.nan)/rc if rc>0 else np.nan
            S.append(r)
    st=pd.DataFrame(S);st.to_csv(OUT/'expanded_boundary_S_metrics.csv',index=False)
    for k,n in enumerate(panel):
        col=names.index(n);x=baseline[:,col].copy();x[~original500]=np.nan
        for mode,y in (('full_release',rate[:,k]),('fixed_fiducials',fixed[:,k])):
            m=vergleichen(x,y);m['icc_a1']=icc(x,y)
            r={'marker_id':n,'mode':mode,'unit':cat.loc[cat.marker_id.eq(n),'unit'].iloc[0],**m}
            if n in keyids:
                ok=np.isfinite(x)&np.isfinite(y);a=x[ok];b=y[ok];boots=[]
                if len(a)>=30:
                    for z in range(1000):
                        ix=rng.integers(0,len(a),len(a));dd=b[ix]-a[ix];mu=dd.mean();sd=dd.std(ddof=1)
                        boots.append([icc(a[ix],b[ix]),mu,mu-1.96*sd,mu+1.96*sd])
                    bs=np.array(boots)
                    for q,key in enumerate(('icc_a1','bias','loa_low','loa_high')):
                        r[key+'_ci_low'],r[key+'_ci_high']=np.nanquantile(bs[:,q],[.025,.975])
                    r['bootstrap_n']=1000;r['bootstrap_unit']='one_ECG_per_person'
            sampling.append(r)
    fs=pd.DataFrame(sampling);fs.to_csv(OUT/'sampling_rate_metrics.csv',index=False)
    # Same frozen 500 cases: matrix comparator for a directly comparable benchmark.
    selection=readp(OUT/'selection_private.parquet',['study_id','split'])
    dower={}
    for p in sorted((VAR/'bloecke').glob('block_*.parquet')):
        f=readp(p,['study_id']+['dower__'+n for n in names]);f=f.loc[f.study_id.isin(selection.study_id)]
        for r in f.to_dict('records'):dower[r['study_id']]=[r['dower__'+n] for n in names]
    dy=np.array([dower[s] for s in selection.study_id],float)
    subset=[]
    for k,n in enumerate(names):
        a=vergleichen(baseline[folds=='train',k],dy[folds=='train',k]);b=vergleichen(baseline[folds=='validation',k],dy[folds=='validation',k])
        subset.append({'marker_id':n,'same500_matrix_class':combine(a['class'],b['class'])})
    nswide=nswide.merge(pd.DataFrame(subset),on='marker_id',validate='one_to_one')
    result=base.merge(nswide,on='marker_id',how='left',validate='one_to_one')
    for mode in ('full_release','fixed_fiducials'):
        z=fs.loc[fs['mode'].eq(mode)].drop(columns=['mode','unit']).rename(columns={k:'fs250_'+mode+'_'+k for k in fs.columns if k not in ('marker_id','mode','unit')})
        result=result.merge(z,on='marker_id',how='left',validate='one_to_one')
    result['boundary4_status']='berechnet_500_EKGs';result.loc[~result.admitted.eq(True),'boundary4_status']='nicht_pruefbar_nicht_zugelassen'
    result['sampling_rate_panel']=result.marker_id.isin(panel)
    # Exclude internal dossier/publication routing fields from the public table.
    public=result.drop(columns=['publikationen_explizit','befunde','publikationen_familie','in_definition','in_all_reference_schemas','aw056_twin','hauptarbeit_B'],errors='ignore')
    public.to_csv(S2,index=False)
    main=result.loc[result.matrix_primary_class.isin(CLASSES)]
    ns_summary={'catalog_counts':counts(result.boundary4_worst_class),'matrix_sensitive_original_set_counts':counts(main.boundary4_worst_class),'same500_matrix_counts':counts(main.same500_matrix_class),'computed_records':int(computed.sum()),'qc_abstentions':int((~computed).sum()),'original500_records':int(original500.sum()),'processing_seconds':seconds}
    summary={'NS05':ns_summary,'S_panel_n':len(panel),'S_family_counts':manifest['panel_family_counts'],'S_n_ecgs':500,'sampling_n_ecgs':int(original500.sum()),'sampling_qc250_eligible':int((original500&qc250).sum()),'sampling_panel_n':len(panel),'sampling_estimable_full':int(fs.loc[fs['mode'].eq('full_release'),'icc_a1'].notna().sum()),'sources':{'manifest_sha256':mhash,'base_sha256':sha(OUT/'matrix_base_aggregate.csv'),'record_hashes':{p.name:sha(p) for p in files}},'report_script_sha256':sha(__file__),'S2_sha256':sha(S2)}
    save('measurement_summary.json',summary)
    FIG.mkdir(parents=True,exist_ok=True)
    # Aggregate-only figure; marker labels and units remain in CSV.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    labels=['Stable','Conditional','Dependent','Constant-identical','Not estimable']
    keys=['stabil','bedingt','transformabhaengig','konstant_identisch','nicht_schaetzbar']
    a=[ns_summary['matrix_sensitive_original_set_counts'].get(k,0) for k in keys]
    ag=readj(OUT/'aggregate_summary.json');b=[ag['calibrated_counts_matrix_set'].get(k,0) for k in keys]
    axes[0].bar(np.arange(5)-.18,a,.36,label='Kors vs Kors ±4 ms (500 ECGs)')
    axes[0].bar(np.arange(5)+.18,b,.36,label='Calibrated Dower vs Kors (5,648 ECGs)')
    axes[0].set_xticks(np.arange(5),labels,rotation=25,ha='right');axes[0].set_ylabel('Markers (original matrix-sensitive set: 1,755)');axes[0].legend(fontsize=7)
    axes[0].set_ylim(0,max(a+b)*1.25)
    ratios=main.matrix_halfwidth_over_rc_gesamt.replace([np.inf,-np.inf],np.nan).dropna()
    axes[1].hist(np.log10(ratios[ratios>0]),bins=35,color='#447799');axes[1].axvline(0,color='black',ls='--');axes[1].set_xlabel('log10(matrix LoA half-width / repeat RC)');axes[1].set_ylabel('Markers');fig.tight_layout()
    fig.savefig(FIG/'AW150_matrix_context.png',dpi=180);fig.savefig(FIG/'AW150_matrix_context.svg');plt.close(fig)
    write_report(result,ag,summary,st,fs)
    print(json.dumps({k:summary[k] for k in ('NS05','S_panel_n','sampling_n_ecgs','sampling_estimable_full')},ensure_ascii=False),flush=True)

def write_report(result,ag,summary,st,fs):
    main=result.loc[result.matrix_primary_class.isin(CLASSES)];dep=main.matrix_primary_class.eq('transformabhaengig');ns=summary['NS05'];fold=ag['fold_concordance'];angle=ag['angle_same_catalog_sample']
    ratios=main.matrix_halfwidth_over_rc_gesamt.replace([np.inf,-np.inf],np.nan).dropna()
    context={'n_finite':len(ratios),'median':float(ratios.median()),'p25':float(ratios.quantile(.25)),'p75':float(ratios.quantile(.75)),'below_one':int((ratios<1).sum())}
    angle_row=result.loc[result.marker_id.eq('spatial_qrs_t_angle_deg')].iloc[0]
    ct=pd.read_csv(OUT/'fold_cross_table.csv').rename(columns={'train_class':'Fold 1'})
    paneltable=fs.loc[fs.marker_id.isin(['spatial_qrs_t_angle_deg','qrs_t_normalized_discrete_frechet','qrs_t_frechet_beat_median','qrs_t_normalized_dtw_distance','qrs_duration_ms','qtc_fridericia_ms']),['marker_id','mode','n_pairs','icc_a1','bias','loa_low','loa_high']].copy()
    for c in ('icc_a1','bias','loa_low','loa_high'):paneltable[c]=paneltable[c].map(lambda v:textnum(v,5))
    byfamily=st.merge(result[['marker_id','familie']],on='marker_id',validate='many_to_one').groupby('familie').agg(markers=('marker_id','nunique'),median_mac_rc=('mac_over_rc_gesamt','median'),max_mac_rc=('mac_over_rc_gesamt','max'),min_pairs=('n_pairs','min')).reset_index()
    recovered=main.loc[dep&main.calibrated_class.isin(['stabil','bedingt'])]
    new_dependent=int((main.matrix_primary_class.eq('bedingt')&main.calibrated_class.eq('transformabhaengig')).sum())
    benchmark_counts=ns['matrix_sensitive_original_set_counts'];benchmark_estimable=sum(benchmark_counts.get(c,0) for c in CLASSES)
    benchmark_stable_fraction=benchmark_counts.get('stabil',0)/benchmark_estimable
    frechet_fs=fs.loc[fs.marker_id.eq('qrs_t_normalized_discrete_frechet')&fs['mode'].eq('full_release')].iloc[0]
    angle_fs=fs.loc[fs.marker_id.eq('spatial_qrs_t_angle_deg')&fs['mode'].eq('full_release')].iloc[0]
    angle_fixed=fs.loc[fs.marker_id.eq('spatial_qrs_t_angle_deg')&fs['mode'].eq('fixed_fiducials')].iloc[0]
    recoverytable=recovered.groupby(['familie','calibrated_class']).size().reset_index(name='Marker')
    doc=f'''# A-Matrixnachtrag AW-150

12.10.2026 · MS/PR · Auftrag AW-150 · Standard 02 Version 1.1 mit Ergänzung 1.2 · deskriptiver technischer Methodenvergleich, keine klinische Zertifizierung.

## Herkunft und Kernreproduktion

Lauf `A155-2026-10-04-d6e70e`. Historischer Katalog: `nachrechnung/transform_katalog.csv`, AW-067. Individuelle Wertepaarquelle: `<local path>` bis `block_028.parquet`, jeweils geprüfter Abschluss-SHA256, Footer und explizite Spaltenprojektion. 5.648 EKGs/Personen, train und validation je 2.824. Auswertung: `werkzeuge/varianten/aw150_analysis.py aggregate`, Quellenhashes: `aw150/aggregate_summary.json`.

**Exakt reproduziert:** 2.005 IDs; 2.004 zugelassen; 1.071 abhängig + 684 bedingt = 1.755 matrixveränderliche, schätzbare Marker; 0 stabil in dieser Gruppe. Separat 217 nichtkonstante identische Ausgaben, 1 konstant-identische Ausgabe, 31 nicht schätzbar und 1 nicht prüfbar. Sämtliche im Reproduktionsvertrag geprüften Zahlen (Paarnenner, Bias, Differenz-SD, LoA, Kors-SD, Spearman, relative LoA-Breite, Quintilwechsel) stimmen mit der Katalog-CSV bis rtol=1e−10/atol=1e−12 überein; Klassen exakt. Dies reproduziert die archivierten Entwicklungswerte, keine Neuberechnung aller Signale. Neue Signalbaseline der Sensitivitäten wurde zusätzlich pro Aufnahme gegen AW-067 geprüft (rtol=1e−9, atol=0, Verfügbarkeit exakt).

## NS-05 - Kalibrierung der Stabilitätsregel, Matrixseite

Vor Ergebnissicht festgelegt: SHA256-Ranking, Seed 20261012, 250 EKGs je Fold aus dem eingefrorenen 5.648er-Sample; insgesamt 500 verschiedene Personen. Keine Auswahl nach Markerwerten. Identische Signale und Release-Grenzen; je getrennt r_onset, r_offset und t_offset ±4 ms an allen vorhandenen Schlägen, sechs Szenarien. Robust-Morphologie bleibt wie in AW-143 unverändert; Release berechnet Marker und Repräsentativgrenzen neu. M5 unverändert: ≥100 Paare je Fold, positive Kors-SD; stabil nur ρ≥0,95, volle LoA-Breite/SD(Kors)≤0,5, Quintilwechsel≤0,10; abhängig bei ρ<0,80 oder Breite>2 oder Wechsel>0,40. Schlechteste Foldklasse je Szenario; zusätzlich konservativer schlechtester Wert über sechs Szenarien. Fehlende Schätzbarkeit wird separat geführt.

{ns['computed_records']} neue Baselines reproduziert; {ns['qc_abstentions']} QC-Abstentionen. Unter den ursprünglich 1.755 matrixveränderlichen Markern: **{ns['matrix_sensitive_original_set_counts']}**. Alle zugelassenen Katalogausgaben: {ns['catalog_counts']}. Dower-Kors auf exakt derselben 500er-Auswahl: {ns['same500_matrix_counts']} innerhalb der ursprünglichen 1.755. Die unveränderte technische Regel kann damit für sehr kleine Grenzverschiebungen sowohl bestanden als auch verfehlt werden; 0/1.755 im vollen Matrixvergleich allein ist kein Äquivalenztest. Kleine Grenzverschiebungen sind ein kontrollierter Nahvergleich, kein bewiesenes klinisch äquivalentes Verfahren. Sparse Marker verlieren bei 250 statt 2.824 EKGs je Fold Schätzbarkeit.

**Einordnung der Regelstrenge:** Nur {benchmark_counts.get('stabil',0)}/{benchmark_estimable} schätzbare Nahvergleichsmarker ({benchmark_stable_fraction:.3%}) bestehen die Stabilitätsregel in allen sechs Grenzrichtungen. Sie ist somit für diesen konservativen Grenzbenchmark streng; eine große Zahl fällt selbst bei derselben Matrix durch. Gleichzeitig sind {benchmark_counts.get('stabil',0)} stabile Marker ein positiver Machbarkeitsbeleg für die Regel. Die {benchmark_counts.get('transformabhaengig',0)} abhängigen Nahvergleichsmarker zeigen Grenz-/Implementierungssensitivität; diese Klasse darf hier nicht als Effekt eines Matrixwechsels bezeichnet werden. Die kleinere Stichprobe verändert auch die Dower-Kors-Klassen (1.194 abhängig statt 1.071 in der vollen Stichprobe); beide Sensitivitäten müssen mit ihren Nennern berichtet werden.

Beleg: `aw150/ns05_boundary4_metrics.csv`, alle IDs, sechs Szenarien, beide Folds und Pool; Markeraggregation samt Richtungen/LoA in Supplementary Data S2; `measurement_manifest.json` und `measurement_summary.json`.

**Englischer Textvorschlag:** “We benchmarked the technical rules using identical Kors signals with separate ±4-ms shifts of QRS onset, QRS offset and T offset in 500 frozen development ECGs (250 per fold). Across the original 1,755 matrix-sensitive outputs, the conservative worst-direction classification was {ns['matrix_sensitive_original_set_counts'].get('stabil',0)} stable, {ns['matrix_sensitive_original_set_counts'].get('bedingt',0)} conditional, {ns['matrix_sensitive_original_set_counts'].get('transformabhaengig',0)} dependent, {ns['matrix_sensitive_original_set_counts'].get('konstant_identisch',0)} constant-identical and {ns['matrix_sensitive_original_set_counts'].get('nicht_schaetzbar',0)} not estimable. These rules therefore quantify sensitivity to a technical perturbation rather than clinical equivalence.”

## NS-22 - Lineare Skalenkalibrierung

Je Marker OLS `Kors = Intercept + Slope × Dower`, mit Zentrierung/Skalierung zur numerischen Stabilität. Fold 1 schätzt ausschließlich für Fold 2, Fold 2 ausschließlich für Fold 1. ≥100 endliche Fitpaare und positive Fitvarianzen. In den Auswertungsfolds unveränderte M5-Regeln, in nativer Einheit; keine In-Sample-Restfehler als Validierung. Pool der beiden out-of-fold Vorhersagen nur deskriptiv. Geschätzte Steigung/Achsenabschnitt, Fitnenner, Target-LoA und Klassen vollständig in S2. Binäre/gebundene Größen werden rechnerisch mitgeführt, die lineare Umrechnung ist dort keine fachliche Rekodierungsvalidierung. Konstante und nicht zugelassene Marker erhalten keine erfundenen Fits.

Nach Umrechnung in der ursprünglichen 1.755er-Gruppe: **{ag['calibrated_counts_matrix_set']}**. Von 1.071 zuvor abhängigen Markern werden **{ag['dependent_to_conditional']} bedingt, {ag['dependent_to_stable']} stabil, {ag['dependent_to_unestimable']} nicht schätzbar**; umgekehrt werden **{new_dependent} zuvor bedingte Marker abhängig**. Damit steigt der abhängige Gesamtnenner um eins. “Nur kalibrierbarer Skalenunterschied” wäre für den Wechsel zu bedingt zu stark: Restabweichung bleibt. Die Zahl vollständig regelstabiler ehemals abhängiger Marker ist {ag['dependent_to_stable']}. Positive affine Umrechnung verändert Rangordnung und Quintilzuordnung theoretisch nicht; sie kann die LoA-Breite verbessern, aber schlechte Rangübereinstimmung nicht reparieren. Negative Steigungen sind ausdrücklich sichtbar und keine reine positive Eichung. Bei periodischen Koordinaten ist eine gewöhnliche affine Regression durch den Koordinatenschnitt begrenzt; Roh- und Wrapped-LoA des historischen Katalogs bleiben getrennt sichtbar.

{table(recoverytable,['familie','calibrated_class','Marker'])}

Beleg: `aw150/matrix_base_aggregate.csv`, `aggregate_summary.json`, S2-Spalten `*_fit_*`, `*_calibrated_*`, `calibration_interpretability`.

**Englischer Textvorschlag:** “Cross-fitted affine calibration, fitted in one development fold and evaluated in the other, changed {ag['dependent_to_conditional']} of 1,071 dependent outputs to conditional and {ag['dependent_to_stable']} to stable under the unchanged rules; {new_dependent} initially conditional outputs became dependent. A conditional reclassification indicates reduced scale-related disagreement, not complete interchangeability. Positive affine calibration preserves ranks and cannot remove rank discordance.”

## NS-35 - Foldkonkordanz und schlechterer Fold

Primärnenner 1.755; identische/konstante und nichtschätzbare Marker separat. Gleiche Klasse: **{fold['same']}/{fold['n']} = {fold['agreement']:.4%}**, ungewichtetes Cohen-κ **{fold['kappa']:.6f}**. Train: {fold['train_counts']}; validation: {fold['validation_counts']}. {fold['dependent_both']} Marker abhängig in beiden Folds, **{fold['dependent_one_only']} nur in einem**. Letztere erhalten die Gesamtklasse durch die konservative Worst-Fold-Regel; das belegt Schwelleninstabilität, keinen systematischen Messbias. Gepoolte Sensitivität: {fold['pool_counts']}; {fold['worse_dependent_pool_not_dependent']} der 1.071 Worst-Fold-abhängigen sind gepoolt nicht abhängig. Pool ist keine unabhängige Prüfung.

{table(ct,list(ct.columns))}

Beleg: `aw150/fold_cross_table.csv` und `aggregate_summary.json`; Rohklassen in S2. κ deskriptiv über korrelierte Katalogmarker, ohne inferentielles KI.

**Englischer Textvorschlag:** “Fold classifications agreed for {fold['same']}/1,755 outputs ({fold['agreement']:.1%}; Cohen’s κ={fold['kappa']:.3f}). Of 1,071 outputs classified as dependent by the worse-fold rule, {fold['dependent_one_only']} were dependent in only one fold. Pooling the 5,648 ECGs classified {fold['pool_counts'].get('transformabhaengig',0)} outputs as dependent; this is a descriptive sensitivity analysis.”

## NS-06 und NS-19 - Matrix- und Grenzabweichung relativ zur Wiederholung

Wiederholungsquelle ausschließlich `rv_a1/repeat_marker_metrics.csv`, populations=`gesamt` bzw. `proxy`. AW-149-Ergebnisse wurden nicht vor deren Abgabe verwendet. Je Marker Matrix-LoA-Halbweite/RC = `1,96×SD(Dower−Kors)/RC`, zusätzlich Differenz-SD/Wiederholungsdifferenz-SD sowie /zwischenpersonaler SD. Nur endliche positive Referenznenner; fehlt RC, bleibt Verhältnis leer. Kurzfristige klinische Wiederholung enthält Zustandsänderung und ist kein isoliertes Messrauschen. Die Matrix- und Wiederholungssamples sind verschieden; Quotienten liefern Größenordnung, keine gepaarte Varianzzerlegung.

Für {context['n_finite']} der 1.755 Marker mit positivem endlichem Gesamt-RC: Median Halbweite/RC **{context['median']:.4f}**, IQR {context['p25']:.4f}-{context['p75']:.4f}; {context['below_one']} unter 1. Flächenwinkel auf der 5.648er-Stichprobe: Matrixhalbweite {1.96*angle['difference_sd']:.4f}°, Gesamt-RC {angle_row.repeat_gesamt_rc:.4f}°, Verhältnis {angle_row.matrix_halfwidth_over_rc_gesamt:.4f}; Proxy-RC {angle_row.repeat_proxy_rc:.4f}°, Verhältnis {angle_row.matrix_halfwidth_over_rc_proxy:.4f}.

B-Störtest: alle sechs Marker × sechs S-Grenzrichtungen aus `nachrechnung/b_stoertest.csv` an RC angeschlossen (`aw150/b_boundary_over_rc.csv`), Kennzahl mittlere absolute Änderung/RC, ausdrücklich **nicht** SD/RC. Erweiterung: {summary['S_panel_n']} Marker, mindestens zehn pro Katalogfamilie, Seed 20261012, dieselben 500 EKGs. Sechs einzelne Störungen ±13,5/11,6/32,8 ms, realisiert ±14/12/32 ms am 500-Hz-Raster. Paarweise Verfügbarkeit, Bias, LoA, mittlere/P95 absolute Änderung und RC-Verhältnisse in `expanded_boundary_S_metrics.csv`. Keine gemeinsame oder empirisch korrelierte Grenzfehlersimulation.

{table(byfamily,['familie','markers','median_mac_rc','max_mac_rc','min_pairs'])}

**Englischer Textvorschlag:** “In {context['n_finite']} matrix-sensitive outputs with an available positive repeat RC, the median ratio of matrix LoA half-width to repeat RC was {context['median']:.3f} (IQR {context['p25']:.3f}-{context['p75']:.3f}). For the area angle the ratio was {angle_row.matrix_halfwidth_over_rc_gesamt:.3f}. These ratios compare different development samples and include clinical change in the repeat denominator; they do not isolate instrument noise. The fixed single-boundary LUDB-SD stress test was extended to {summary['S_panel_n']} outputs across all 11 catalog families in 500 ECGs.”

## NS-20 - Abtastrate 500 gegen 250 Hz

Vor Ergebnissicht gebundener Filter: 81-Tap FIR, Kaiser β=8,6, Grenzfrequenz 112,5 Hz bei fs=500; `scipy.signal.resample_poly(up=1,down=2,axis=0,window=taps,padtype='constant')`, lineare FIR-Phase durch SciPy kompensiert. Koeffizienten und Hash im Manifest. Identische physikalische Aufnahme; nur originär 500-Hz-Aufnahmen zulässig. Nach Rohsignal-Dezimation unveränderte Release-Vorverarbeitung, R-Erkennung, Fusion, Morphologiekonstruktion und Registry bei fs=250; das ist ein Test der gesamten Messkette, nicht nur geometrischer Diskretisierung. Zusätzliche feste, auf 250 Hz gerundete Fiducials trennen einen Anteil der Neudelineation ab. Filterdämpfung oberhalb der neuen Nyquistfrequenz ist Teil des beabsichtigten Eingriffs; kein reiner Rastereffekt.

{summary['sampling_n_ecgs']} originäre 500-Hz-EKGs, davon {summary['sampling_qc250_eligible']} nach unverändertem Release-Rohsignal-QC bei 250 Hz zugelassen; {summary['sampling_panel_n']} Marker ausgewählt; {summary['sampling_estimable_full']} ICC-Schätzungen im vollen Releasevergleich. Das Rohsignal-QC wurde mit `repair_short_gaps`/`assess_signal_quality` zusätzlich bei 250 Hz angewandt; die vorab berechneten Marker werden bei QC-Abstention nicht als gültig gezählt. Das Qualitätsobjekt der geladenen 500-Hz-Aufnahme geht nicht in die Registry-Markerrechnung ein. ICC(A,1), Bias und normaltheoretische LoA je Marker; sechs Schlüsselmarker mit 1.000 Personenbootstrapziehungen, Punktintervalle für ICC/Bias/jede LoA-Grenze, Seed im Manifest; keine simultanen Intervalle. Nichtschätzbarkeit und Verfügbarkeit in CSV, keine Ersatzwerte.

{table(paneltable,list(paneltable.columns))}

**Einordnung:** Die diskrete Fréchet-Kopplung erreicht im vollen Vergleich ICC {frechet_fs.icc_a1:.5f} bei {int(frechet_fs.n_pairs)} Paaren; ihre LoA betragen {frechet_fs.loa_low:.5f} bis {frechet_fs.loa_high:.5f} in nativer Einheit. Der Flächenwinkel erreicht nur ICC {angle_fs.icc_a1:.5f} bei {int(angle_fs.n_pairs)} Paaren, LoA {angle_fs.loa_low:.5f}° bis {angle_fs.loa_high:.5f}°. Mit festen Fiducials beträgt der Winkel-ICC {angle_fixed.icc_a1:.5f}; dennoch bleiben LoA {angle_fixed.loa_low:.5f}° bis {angle_fixed.loa_high:.5f}°. Eine geringe mittlere Verschiebung verdeckt somit große individuelle Streuung. Das Panel liefert keinen allgemeinen Nachweis der Abtastrateninvarianz. Unterschiede zwischen voller und fester Delineation sind wegen markerweise verschiedener endlicher Paare keine kausale Varianzzerlegung.

Beleg: `aw150/sampling_rate_metrics.csv` (500-Hz-Baseline gegen 250-Hz-Wert, Differenz 250−500), S2-Spalten `fs250_*`; unveränderte Releasequellen im Manifest gebunden.

**Englischer Textvorschlag:** “A controlled sampling-rate sensitivity analysis compared {summary['sampling_n_ecgs']} identical 500-Hz recordings with their anti-aliased 250-Hz versions using an 81-tap Kaiser FIR filter (112.5-Hz cutoff) and unchanged release algorithms at the respective sampling rates. Discrete Fréchet coupling had ICC(A,1) {frechet_fs.icc_a1:.3f} ({int(frechet_fs.n_pairs)} pairs; LoA {frechet_fs.loa_low:.3f} to {frechet_fs.loa_high:.3f}), whereas the area angle had ICC {angle_fs.icc_a1:.3f} ({int(angle_fs.n_pairs)} pairs; LoA {angle_fs.loa_low:.2f}° to {angle_fs.loa_high:.2f}°). Full re-delineation and fixed-fiducial results for {summary['sampling_panel_n']} outputs are distinguished in Supplementary Data S2; these findings do not establish general sampling-rate invariance.”

## NS-21 - Einheitliche Flächenwinkel-Stichprobe

Die historische 11.529er-Trainauswahl und die angereicherte 5.648er-Auswahl sind nicht derselbe Nenner. Historische Kernzahlen werden nicht zur Begründung der 5.648er-Katalogklassen verwendet. Neu auf derselben 5.648er-Aufnahmeliste: **{angle['n_pairs']} endliche Flächenwinkelpaare**, Spearman **{angle['spearman']:.6f}**, Bias **{angle['bias']:.6f}°**, LoA **{angle['loa_low']:.6f}° bis {angle['loa_high']:.6f}°**, >105°-Wechsel **{angle['thresholds']['105']['switch_n']}/{angle['n_pairs']} = {angle['thresholds']['105']['switch_fraction']:.6%}**. Beide Verfahren >105° zählen strikt; Gleichheit gilt als nicht darüber. 90°/120°-Sensitivitäten ebenfalls in `aggregate_summary.json`. Diese Werte ersetzen bei Übernahme in R3/D3 die historischen Werte der Trainauswahl; historische Intervalle dürfen nicht angehängt werden. Keine klinische Schwellenvalidierung.

**Englischer Textvorschlag:** “On the same frozen 5,648-record catalog sample, {angle['n_pairs']} finite area-angle pairs showed Spearman {angle['spearman']:.3f}, bias {angle['bias']:+.3f}° and LoA {angle['loa_low']:+.3f}° to {angle['loa_high']:+.3f}°. Substituting inverse Dower for Kors switched the >105° stress-test label in {angle['thresholds']['105']['switch_n']}/{angle['n_pairs']} pairs ({angle['thresholds']['105']['switch_fraction']:.3%}). These estimates supersede the historical training-subsample example for comparison with Table 3.”

## NS-01 - Zusammenführbare Markerdatei

`publikationen/PUB-01_messvaliditaet/supplementary_data_S2_marker_matrix.csv`: genau 2.005 eindeutige unveränderte `marker_id`, Familie und Einheit; ursprüngliche Klassen/LoA/Verfügbarkeit je Fold und Pool; getrennte Invarianz; Crossfitparameter und Zielklassen/LoA; RC-Verhältnisse gesamt/Proxy; ±4-ms-Richtungen und Klassen/LoA; 250-Hz-Auswahl und Kennzahlen. Merge mit AW-149 ausschließlich über Marker-ID, nie über Zeilenposition. Fehlende Ergebnisse für Nichtzulassung/Nichtschätzbarkeit/Panelnichtzugehörigkeit bleiben leer mit Statusfeldern. Nur Markeraggregate, keine Personen- oder EKG-Kennungen. Erweiterte S- und Abtastraten-CSV werden zusätzlich nach `abbildungen/v11/` als veröffentlichbare Aggregatbelege kopiert.

**Englischer Textvorschlag:** “Supplementary Data S2 provides marker-level fold classes, native-unit LoA, cross-fitted calibration parameters and results, ratios to clinical repeat RC, and controlled boundary and sampling-rate sensitivity results. The matrix and reliability tables share unchanged marker identifiers and contain no patient-level records.”

## Prüfkette, Auflagen und Widerlegbarkeit

P01a bestanden mit Auflage: aktuelle AW-067-Blockhashes und gebundener Release geprüft; historische vollständige Erzeugeridentität wird nicht rückwirkend attestiert. P01b bestanden: Kernzählungen und alle geprüften Metriken reproduziert. P01c bestanden für die {ns['computed_records']} neu berechneten Baselines aus 500 ausgewählten Aufnahmen ({ns['qc_abstentions']} QC-Abstentionen), kein universeller Vollkatalogsignal-Reproduktionsclaim. P02/P03: technischer Vergleich, Nenner und Einheiten getrennt. P05: nur train/validation; gesperrte Testmenge weder geöffnet noch aufgelistet. P09: vollständige Paarnenner, Schätzbarkeit und Panelindikator. P16/P17: kontrollierte Release-Signalvarianten, keine neue klinische Referenz. P08: Schlüssel-Abtastratenintervalle personenbasiert; Katalogklassen/κ/Quotienten/S-Störung ohne Patientenbootstrap **offen** für inferentielle Aussagen. RC/LoA sind Streuungsgrenzen, keine Konfidenzintervalle. P23: keine Recherche laut Auftrag, keine neue Literaturvalidierung. P25: vollständige Aggregatdateien und reproduzierbare Skripte; K-Abnahme ausstehend.

| Falle | Stellungnahme für alle NS-Befunde |
|---|---|
| F1 | Nicht zutreffend: keine Marker-Endpunkt-Assoziation; technische Regelklassen sind definitionsabhängig. |
| F2 | Eingeschränkt: angereicherte eingefrorene Entwicklungsstichprobe, keine Populationsprävalenz/PPV/NPV. |
| F3 | Eingeschränkt: feste M5-Regel, prospektiv gebundene neue Panel-/Filterwahl; OLS strikt foldübergreifend, keine Auswahl günstiger Fits. |
| F4 | Eingeschränkt: vollständiger Katalog und alle gewählten Panelmarker sichtbar; keine Gewinner-/Signifikanzbehauptung, Intervalle punktweise. |
| F5 | Eingeschränkt/offen: paarweise endliche Werte, keine Missingness-Imputation; kleine Sensitivitätsstichprobe verschärft Sparse-Marker-Ausfall. |
| F6 | Eingeschränkt: Kors als technischer Anker, kein unabhängiger Goldstandard; Nahvergleich prüft Regelstrenge, keine klinische Äquivalenz. |
| F7 | Eingeschränkt: eine Aufnahme je Person im Matrixvergleich; RC stammt aus anderen klinischen Wiederholungen, keine Zustands-/Rauschzerlegung. |
| F8 | Eingeschränkt: identische Aufnahmen/Kontexte, kontrollierte Grenz-/Filteränderung; Veränderungen sind technische Nichtinvarianz, kein Krankheitsbefund. |

Widerlegbarkeit: andere repräsentative Akquisitionspopulation, andere zulässige klinische Toleranzen, nichtlineare/familiengerechte Kalibrierung oder zustandskontrollierte Wiederholungs-RC können die technische Einordnung verändern. Keine dieser Möglichkeiten macht die aktuellen Entwicklungswerte zu extern bestätigten Ergebnissen. Lücken/Chancen: kategoriale Rekodierung gesondert prüfen, periodische Descriptoren nicht als gewöhnliche lineare Eichung interpretieren, Clusterunsicherheit für Klassen und Verhältnisse separat beauftragen. Eine Worst-Direction-Klasse ist konservativer als ein einzelner Grenzvergleich. Kein Manuskript direkt geändert; Textübernahme und Fachstatus entscheidet K.

## Dateien und Ausführung

Neue Werkzeuge: `aw150_analysis.py`, `aw150_report.py`, `aw150_start.ps1` ausschließlich unter `werkzeuge/varianten/`. Phasen `aggregate`, `prepare`, `measure`, `report`, `verify`; Start über `aw150_start.ps1 -Phase <Phase>`, Python ausschließlich Auswertungsvenv, `-B`, numerische Threads=1. Losgelöster eigener Prozess, PID protokolliert; nur ein Pythonprozess zur selben Zeit. Vor jedem Start RAM und Commit ≥1,5 GiB; Ressourcenwächter begrenzt eigenen RSS/Commit auf 1,5 GiB. Nach jeder Aufnahme gehashte Manifestbindung und atomarer Checkpoint unter `<local path>`. Keine Installation, kein fremder Prozesseingriff, keine Release-/Analyse-/Manuskript-/Registeränderung.

Sicherheit: hoch für archivierte Zahlenreproduktion, Foldtrennung und technische Nichtinvarianz auf den geprüften Aufnahmen; begrenzt für populationsweite Robustheit, klinische Äquivalenz und inferentielle Katalogzählungen. Quellenfehler-Kandidaten: keine fremde Literatur geprüft; keine behaupteten Quellenfehler. Offene Manuskriptplatzhalter: keine neuen Manuskripte, keine Autor-/Ethik-/COI-Angaben bearbeitet.
'''
    (PROJECT/'nachrechnung/A_MATRIX_NACHTRAG_AW150.md').write_text(doc,encoding='utf8')
    for filename in ('expanded_boundary_S_metrics.csv','sampling_rate_metrics.csv','b_boundary_over_rc.csv','fold_cross_table.csv'):
        (FIG/('AW150_'+filename)).write_bytes((OUT/filename).read_bytes())
    save('report_context.json',context)
    delivery=f'''# Abgabe AW-150

## Ergebnis

Auftrag rechnerisch vollständig bearbeitet, K-Abnahme und Manuskriptübernahme ausstehend. Kernzählung exakt reproduziert: 1.071 abhängig, 684 bedingt, 0 stabil unter 1.755 matrixveränderlichen Markern; 217 nichtkonstante identische und 1 konstante identische Ausgabe separat.

- NS-05: Kors/Kors mit sechs ±4-ms-Grenzverschiebungen auf 500 Entwicklungs-EKGs; Klassen unter den ursprünglichen 1.755: {ns['matrix_sensitive_original_set_counts']}. Nichtschätzbarkeit separat.
- NS-22: out-of-fold OLS-Umrechnung für jeden zugelassenen schätzbaren Marker; von 1.071 abhängig werden {ag['dependent_to_conditional']} bedingt und {ag['dependent_to_stable']} stabil, umgekehrt werden {new_dependent} bedingte Marker abhängig. Kein Beleg vollständiger Kalibrierbarkeit aus einem Wechsel zu bedingt.
- NS-35: Klassenkonkordanz {fold['same']}/{fold['n']} ({fold['agreement']:.3%}), κ={fold['kappa']:.6f}; {fold['dependent_one_only']} nur durch einen abhängigen Fold insgesamt abhängig. Gepoolte Sensitivität vollständig.
- NS-06/19: Matrix-LoA-Halbweite/RC je Marker, B-Störung/RC und Erweiterung S auf {summary['S_panel_n']} Marker (≥10 je Familie), 500 Personen, Seed fest.
- NS-20: anti-aliased 500/250-Hz-Vergleich auf {summary['sampling_n_ecgs']} originären 500-Hz-Aufnahmen, {summary['sampling_panel_n']} Marker inklusive Fréchet; voller Release und feste Fiducials getrennt, ICC/LoA; sechs Schlüsselmarker mit Bootstrap-KI.
  Diskrete Fréchet-Kopplung ICC {frechet_fs.icc_a1:.5f} ({int(frechet_fs.n_pairs)} Paare), Flächenwinkel ICC {angle_fs.icc_a1:.5f} ({int(angle_fs.n_pairs)} Paare), Winkel-LoA {angle_fs.loa_low:.5f}° bis {angle_fs.loa_high:.5f}°; allgemeine Abtastrateninvarianz nicht belegt.
- NS-21: einheitliche 5.648er-Liste, {angle['n_pairs']} endliche Winkelpaare; ρ={angle['spearman']:.6f}, Bias={angle['bias']:.6f}°, LoA {angle['loa_low']:.6f}° bis {angle['loa_high']:.6f}°; >105°-Wechsel {angle['thresholds']['105']['switch_n']}/{angle['n_pairs']} ({angle['thresholds']['105']['switch_fraction']:.6%}).
- NS-01: veröffentlichbare, zusammenführbare S2-Matrix-CSV für genau 2.005 IDs.

## Neue/geänderte Dateien

- `nachrechnung/A_MATRIX_NACHTRAG_AW150.md`: Ergebnisse je NS-Punkt, Herkunft, Grenzen, F1-F8 und englische Textvorschläge.
- `publikationen/PUB-01_messvaliditaet/supplementary_data_S2_marker_matrix.csv`: Markeraggregate.
- `publikationen/PUB-01_messvaliditaet/abbildungen/v11/AW150_*`: eine Aggregatabbildung PNG/SVG sowie vier Aggregat-CSV.
- `werkzeuge/varianten/aw150_analysis.py`, `aw150_report.py`, `aw150_start.ps1`: neue gekapselte Werkzeuge.
- `<local path>`: private Auswahl und Messcheckpoints, Ressourcen-, Quellen- und Abschlussbelege; Personenbezogenes ausschließlich dort.
- `koordination/abgaben/AW-150.md`: diese Abgabe.

## Belege/Befehle

`werkzeuge/varianten/aw150_start.ps1 -Phase aggregate`, `-Phase prepare`, `-Phase measure -Limit 1` (Pilot), `-Phase measure` (Fortsetzung), `-Phase report`, `-Phase verify`. Detailquellen: AW-067 abgeschlossene 29 Wertepaarblöcke, `nachrechnung/transform_katalog.csv`; A-RC ausschließlich `rv_a1/repeat_marker_metrics.csv` populations gesamt/proxy. Keine vorab verwendeten AW-149-Zwischenergebnisse.

`aggregate_summary.json`: Kern- und Einzelzahlenabgleich, Quellenhashes und Skripthash. `measurement_manifest.json`: Auswahl, Familie/Panel, Grenzen, Filterkoeffizienten/Hash, Releasebindung vor Ergebnissicht. `measurement_summary.json`: {ns['computed_records']} äquivalente Baselines, {ns['qc_abstentions']} QC-Abstentionen, alle 500 Aufnahmecheckpoints mit SHA256. Marker-/Fold-/Richtungsbelege in S2 und privaten Aggregatdateien, Prüfung durch `verify` separat in `verification.json`.

## Offene Fragen / Auflagen

K: englische Vorschläge in M5/R3/R4/Table 3/D3/S2 übernehmen; historische 11.529er-Werte und deren KI nicht mit den neuen 5.648er-Werten mischen. “Pending sampling-rate analysis” durch den begrenzten Panelbefund ersetzen; keine globale Abtastrateninvarianz behaupten. Klassen/κ/Quotienten/S-Störung sind deskriptiv ohne Patientenbootstrap, inferentielles P08-Gate dafür offen. Sparse Marker sind auf 250 Fällen je Fold teilweise nicht schätzbar. Klinische Äquivalenz und externe Prüfung bleiben außerhalb des Auftrags. Lineare Fits für Kategorien/periodische Koordinaten sind ausschließlich technische Sensitivitäten.

## Nebenbefunde und Quellenfehler-Kandidaten

Schlechter-Fold-Regel und Kalibrierung ändern technische Klassen, ohne die Messwerte klinisch zu validieren. Wiederholungs-RC enthält Zustandsänderung; Verhältnis zur Matrixhalbweite ist keine reine Rauschzerlegung. Der 500/250-Hz-Test enthält Anti-Aliasing und Neudelineation; feste Fiducials als Sensitivität separat. Keine Recherche und keine behaupteten fremden Quellenfehler. Kein Manuskript neu erstellt, keine neuen Standardangaben-Platzhalter.

## Sicherheit und Ressourcen

Hoch für Zahlenreproduktion, Foldtrennung, Release-/Checkpointbindung und kontrollierte Unterschiede. Begrenzte Aussage zur Populationsübertragbarkeit und klinischen Austauschbarkeit. Vor jedem Pythonstart ≥1,5 GiB RAM und Commit; ein eigener Prozess mit ≤1,5 GiB Budget, numerische Threads=1, losgelöst/checkpointbasiert. Start- und Endressourcen pro Phase auf E:. Kein fremder Prozess beendet, nichts gelöscht/installiert, keine Außenkommunikation, Testpartition unberührt, nur erlaubte neue Dateien und Markeraggregate im Projekt.
'''
    (PROJECT/'koordination/abgaben/AW-150.md').write_text(delivery,encoding='utf8')

def verify():
    result=pd.read_csv(S2);summary=readj(OUT/'aggregate_summary.json');measurement=readj(OUT/'measurement_summary.json')
    assert len(result)==2005 and result.marker_id.is_unique
    assert counts(result.matrix_primary_class)==summary['primary_counts']
    assert set(pd.read_csv(RC).marker_id)<=set(result.marker_id)
    assert counts(result.loc[result.matrix_primary_class.isin(CLASSES),'calibrated_class'])==summary['calibrated_counts_matrix_set']
    assert sha(S2)==measurement['S2_sha256']
    assert measurement['NS05']['computed_records']+measurement['NS05']['qc_abstentions']==500
    ns=pd.read_csv(OUT/'ns05_boundary4_metrics.csv');assert len(ns)==2004*6*3
    assert not any(k in result.columns for k in ('subject_id','study_id','waveform_path','ecg_time'))
    fs=pd.read_csv(OUT/'sampling_rate_metrics.csv');st=pd.read_csv(OUT/'expanded_boundary_S_metrics.csv');panel=readj(OUT/'measurement_manifest.json')['panel_marker_ids']
    assert len(fs)==len(panel)*2 and len(st)==len(panel)*6
    assert set(fs.marker_id)==set(st.marker_id)==set(panel)
    # Meaningful numerical identities and failure visibility.
    assert np.allclose(result.pool_loa_high-result.pool_loa_low,3.92*result.pool_difference_sd,equal_nan=True)
    good=fs.difference_sd.notna();assert np.allclose(fs.loc[good,'loa_high']-fs.loc[good,'loa_low'],3.92*fs.loc[good,'difference_sd'])
    assert np.allclose(fs.icc_a1.dropna().clip(-1,1),fs.icc_a1.dropna())
    assert icc(np.arange(100.),np.arange(100.))==1
    assert abs(icc(np.arange(100.),np.arange(100.)+10)-.9444444444444444)<.01
    csvs=[S2,*FIG.glob('AW150_*.csv')]
    for p in csvs:
        f=pd.read_csv(p);assert not set(f.columns)&{'subject_id','study_id','waveform_path','ecg_time'}
    outputs=[S2,PROJECT/'nachrechnung/A_MATRIX_NACHTRAG_AW150.md',PROJECT/'koordination/abgaben/AW-150.md',*FIG.glob('AW150_*')]
    save('verification.json',{'status':'passed','checks':['2005 unique marker IDs','core reproduction','crossfit classes','500 complete checkpoints','2004x6x3 benchmark rows','panel complete','native-unit LoA identities','ICC selfcheck','aggregate-only project exports'],'outputs':{str(p):sha(p) for p in outputs},'report_script_sha256':sha(__file__)})
    print('VERIFY passed',flush=True)
