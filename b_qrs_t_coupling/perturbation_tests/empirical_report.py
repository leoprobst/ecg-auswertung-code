"""E-Aggregation, Clusterbootstrap und Ablösung des historischen Stopberichts."""
from werkzeuge.b_stoertest.empirical import OUT, PROJECT, IDS, SEED, SPEC, SPEC_HASH, ADD, ADD_HASH, sha, save, errors
import csv, json, datetime

def rho_draws(baseline, values):
    import numpy as np
    from scipy.stats import rankdata
    valid=np.isfinite(values)&np.isfinite(baseline[:,None,:])
    x=np.broadcast_to(baseline[:,None,:],values.shape)
    # Ungültige Paare am Rangende; endliche Paar-Ränge sind dadurch unverändert.
    rx=rankdata(np.where(valid,x,np.inf),axis=0,method='average')
    ry=rankdata(np.where(valid,values,np.inf),axis=0,method='average')
    n=valid.sum(axis=0);mean=(n+1)/2
    dx=np.where(valid,rx-mean,0);dy=np.where(valid,ry-mean,0)
    with np.errstate(divide='ignore',invalid='ignore'):
        rho=(dx*dy).sum(axis=0)/np.sqrt((dx*dx).sum(axis=0)*(dy*dy).sum(axis=0))
    rho[n<3]=np.nan
    return rho,n

def load():
    import numpy as np
    status=json.loads((OUT/'E_status.json').read_text(encoding='utf8'))
    assert status['status']=='E_complete' and status['done']==500
    manifest=json.loads((OUT/'E_run_manifest.json').read_text(encoding='utf8'))
    for k,p in [('selection_sha256',OUT/'selection_private.parquet'),('script_sha256',PROJECT/'werkzeuge/b_stoertest/empirical.py'),('adapter_sha256',PROJECT/'werkzeuge/b_stoertest/run.py'),('nachtrag_sha256',ADD),('errors_audit_sha256',OUT/'E_errors_audit.json'),('shifts_sha256',OUT/'E_shifts_private.npy')]:
        assert manifest[k]==sha(p)
    paths=sorted((OUT/'records_E').glob('*.json'))
    assert len(paths)==500
    shifts=np.load(OUT/'E_shifts_private.npy')
    # study_id ist im Dateinamen; numerische Ordnung wie im Ziehungsrezept.
    paths.sort(key=lambda p:int(p.stem))
    baseline=[];values=[];checkpoint_hashes={}
    for i,p in enumerate(paths):
        r=json.loads(p.read_text(encoding='utf8'))
        assert r['equivalent'] and len(r['E'])==20
        assert all(manifest[k]==v for k,v in r['binding'].items())
        assert np.array_equal(np.array([x['samples'] for x in r['E']]),shifts[i])
        s=json.loads((OUT/'records'/p.name).read_text(encoding='utf8'))
        assert all(s['equivalent']) and np.isclose(r['baseline'],s['baseline'],rtol=1e-8,atol=0).all()
        baseline.append(r['baseline']);values.append([x['values'] for x in r['E']]);checkpoint_hashes[p.name]=sha(p)
    save('E_checkpoint_hashes_private.json',checkpoint_hashes)
    return np.array(baseline,float),np.array(values,float)

def estimates(baseline,values):
    import numpy as np
    delta=values-baseline[:,None,:]
    valid=np.isfinite(delta);count=valid.sum(axis=(0,1))
    total=np.nansum(delta,axis=(0,1));squares=np.nansum(delta*delta,axis=(0,1))
    sd=np.sqrt((squares-total*total/count)/(count-1))
    rho,n=rho_draws(baseline,values)
    return sd,np.nanmedian(rho,axis=0),rho,n

def report():
    import numpy as np
    baseline,values=load()
    sd,rho,draw_rho,pair_n=estimates(baseline,values)
    delta=values-baseline[:,None,:]
    moments=[np.isfinite(delta).sum(axis=1),np.nansum(delta,axis=1),np.nansum(delta*delta,axis=1)]
    rng=np.random.default_rng(SEED);boot=np.empty((1000,2,6))
    for b in range(1000):
        ix=rng.integers(0,500,500)
        n,s,q=[a[ix].sum(axis=0) for a in moments]
        boot[b,0]=np.sqrt((q-s*s/n)/(n-1))
        boot[b,1]=np.nanmedian(rho_draws(baseline[ix],values[ix])[0],axis=0)
        if (b+1)%100==0:print(json.dumps({'bootstrap_done':b+1}),flush=True)
    lo,hi=np.nanpercentile(boot,[2.5,97.5],axis=0)
    repeat=OUT.parent/'rv_a1/repeat_marker_metrics.csv'
    original=json.loads((OUT/'abschluss.json').read_text(encoding='utf8'))
    assert sha(repeat)==original['hashes'][str(repeat)]
    with repeat.open(encoding='utf8') as f:
        denominator_rows=[r for r in csv.DictReader(f) if r['population']=='gesamt' and r['marker_id'] in IDS]
    assert len(denominator_rows)==6
    for r in denominator_rows:assert np.isclose(float(r['sem']),float(r['difference_sd'])/np.sqrt(2),rtol=1e-12,atol=0)
    den={r['marker_id']:float(r['sem']) for r in denominator_rows}
    assert len(den)==6 and all(v>0 for v in den.values())
    e=[]
    for j,k in enumerate(IDS):
        ratio=sd[j]/den[k]
        cat='grenzrobust relativ zur Kurzzeitstreuung' if ratio<.5 else 'mäßig grenzempfindlich' if ratio<1 else 'grenzempfindlich'
        valid=np.isfinite(values[:,:,j]);n=int(valid.sum())
        e.append({'marker_id':k,'scenario':'E','n_ecgs':500,'n_draws_per_ecg':20,'n_total_cases':10000,'n_computable':n,'n_not_computable':10000-n,
                  'n_ecgs_with_any_noncomputable':int((~valid).any(axis=1).sum()),'n_ecgs_all_draws_noncomputable':int((~valid).all(axis=1).sum()),
                  'within_subject_sd':den[k],'sd_stoer':float(sd[j]),'R':float(ratio),'rho':float(rho[j]),
                  'sd_ci_low':float(lo[0,j]),'sd_ci_high':float(hi[0,j]),'R_ci_low':float(lo[0,j]/den[k]),'R_ci_high':float(hi[0,j]/den[k]),
                  'rho_ci_low':float(lo[1,j]),'rho_ci_high':float(hi[1,j]),'rho_draws_computable':int(np.isfinite(draw_rho[:,j]).sum()),
                  'rho_pairs_min':int(pair_n[:,j].min()),'rho_pairs_max':int(pair_n[:,j].max()),'status':'berechnet','reason':'','category':cat})
    original_csv=PROJECT/'nachrechnung/b_stoertest.csv'
    # S bleibt exakt unverändert; Backup vor der erstmaligen Ablösung.
    backup=OUT/'b_stoertest_vor_E.csv'
    if not backup.exists():backup.write_bytes(original_csv.read_bytes())
    with backup.open(encoding='utf8') as f:s=[r for r in csv.DictReader(f) if r['scenario']=='S']
    assert len(s)==36
    for r in s:r['category']='nicht anwendbar; Kategorie nur aus E'
    rows=s+e;fields=list(dict.fromkeys(k for r in rows for k in r))
    with original_csv.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    save('E_aggregate.json',{'rows':e,'bootstrap':1000,'seed':SEED,'ddof':1,'rho_definition':'median of 20 draw-specific Spearman correlations; pairwise finite per draw',
          'bootstrap_definition':'500 ECG/person clusters sampled with replacement; all 20 draws retained; recompute pooled SD and draw correlations; percentile 2.5/97.5; fixed repeat denominator',
          'comparison_angle_larger_nonoverlap':e[3]['R_ci_low']>e[0]['R_ci_high'],
          'comparison_primary_larger_nonoverlap':e[0]['R_ci_low']>e[3]['R_ci_high']})
    write_documents(e,s,repeat)

def write_documents(e,s,repeat):
    def fmt(x):return f'{float(x):.6g}'
    def ci(r,key):return f"[{fmt(r[key+'_ci_low'])}; {fmt(r[key+'_ci_high'])}]"
    target=PROJECT/'nachrechnung/B_STOERTEST_AW143.md'
    backup=OUT/'B_STOERTEST_AW143_vor_E.md'
    if not backup.exists():backup.write_bytes(target.read_bytes())
    old=backup.read_text(encoding='utf8')
    s_section=old.split('## Szenario S: mittlere absolute Änderung')[1].split('## Claimgrenzen und F1-F8')[0]
    methods=old.split('## Durchführung und Ergebnis')[1].split('## Szenario E:')[0]
    old_hash_table=old.split('## Quellenbindung')[1].split('| Quelle | SHA-256 |')[1].split('Sicherheit:')[0]
    angle,primary=e[3],e[0]
    if angle['R_ci_low']>primary['R_ci_high']:
        comparison='Das R-Intervall des Winkels liegt vollständig oberhalb des Primärmarkers. Nach §6 ist dies eine zusätzliche Grenze des Claims „über den Winkel hinaus“: Die Kovariate besitzt eine ungleiche Messgüte hinsichtlich der geprüften Grenzstörung.'
    elif primary['R_ci_low']>angle['R_ci_high']:
        comparison='Das R-Intervall des Primärmarkers liegt vollständig oberhalb des Winkels. Die in §6 festgelegte zusätzliche Claimgrenze (Winkel deutlich empfindlicher als Primärmarker) wird damit nicht ausgelöst. Daraus folgt keine Freigabe des Claims „über den Winkel hinaus“ und keine gleiche Messgüte.'
    else:
        comparison='Die R-Intervalle von Primärmarker und Winkel überlappen. Die in §6 festgelegte zusätzliche Claimgrenze (Winkel deutlich empfindlicher mit nicht überlappenden Intervallen) wird nicht ausgelöst. Messgütegleichheit ist dadurch nicht belegt.'
    table=['| Marker | SD_stör [95-%-KI] | within-subject SD | R [95-%-KI] | ρ [95-%-KI] | nicht berechenbar / 10.000 | Kategorie |','| --- | --- | --- | --- | --- | --- | --- |']
    for r in e:table.append(f"| {r['marker_id']} | {fmt(r['sd_stoer'])} {ci(r,'sd')} | {fmt(r['within_subject_sd'])} | {fmt(r['R'])} {ci(r,'R')} | {fmt(r['rho'])} {ci(r,'rho')} | {r['n_not_computable']} | {r['category']} |")
    cats='; '.join(f"`{r['marker_id']}`: {r['category']} (R={fmt(r['R'])})" for r in e)
    robust_sentence=' '.join(f"Bei `{r['marker_id']}` verschieben Grenzfehler die Markerwerte gemäß §6 stärker als die Wiederholungsstreuung (R={fmt(r['R'])})." for r in e if r['R']>=1)
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    from werkzeuge.b_stoertest.run import RELEASE
    files=[ADD,PROJECT/'werkzeuge/b_stoertest/empirical.py',PROJECT/'werkzeuge/b_stoertest/empirical_report.py',PROJECT/'werkzeuge/b_stoertest/start_e.ps1',RELEASE/'core/delineation/base.py',OUT/'E_errors_audit.json',OUT/'E_run_manifest.json',OUT/'E_aggregate.json']
    hashes='\n'.join(f'| {p} | {sha(p)} |' for p in files)
    missing_table='\n'.join(f"| {r['marker_id']} | {r['n_ecgs_with_any_noncomputable']} | {r['n_ecgs_all_draws_noncomputable']} | {r['rho_pairs_min']}-{r['rho_pairs_max']} | {r['rho_draws_computable']} |" for r in e)
    text=f'''# AW-143 - B-Störtest (S und E vollständig)

**500 train-EKGs von 500 Personen; Szenario E mit 20 Zügen und 1.000 Clusterbootstrapziehungen abgeschlossen. Szenario S unverändert übernommen.**

{cats}.

{comparison}

MS/PR · Abschluss UTC {stamp}. Verbindliche Spezifikation v1 (`{SPEC_HASH}`) und Nachtrag 1 (`{ADD_HASH}`) geprüft und identisch. AGENTS.md vollständig gelesen. Keine B-Endpunkte, Labels oder klinischen Referenzdaten, keine Testpartition geladen. Die ausdrücklich freigegebenen LUDB-Grenzfehler wurden als technische Störungsquelle gelesen, ohne Expertenpositionen oder einzelne Referenz-EKGs zu laden. Entwicklungsprüfung; keine Bestätigung in Fold 2 oder unabhängige Prüfung. B-Testzahlen, Primärmodell und Erfolgsregel unverändert.

## Durchführung und Äquivalenz
{methods}
Die S-Checkpoints und das ursprüngliche Adapterskript blieben unverändert. Die Wiederaufnahme E verwendet dieselbe private Auswahl; zusätzlich erneute ungestörte Äquivalenz auf allen 500 EKGs gegen die S-Baselines (rtol=1e-8, atol=0). Alle bestanden. `empirical.py::shifted_all` verschiebt die drei Grenzen gleichzeitig auf einer tiefen Kopie und ruft unveränderte Releasefunktionen auf. Der prozesslokale Adapter-Speicherguard wurde auf die für AW143 geltenden 1,5 GiB gesetzt; keine gemeinsame Quelldatei geändert.

## Nachtrag und empirische Ziehung

S wurde am 12.10.2026 vollständig gerechnet; E wurde damals wegen nicht definierter Binnenklassen-/Tailziehung gestoppt. K präzisierte die Quelle am 12.10.2026 laut Auftrag etwa 21:00 **nach Sicht auf S und vor jedem E-Ergebnis**. Diese Reihenfolge wird offengelegt; die Gesamtprüfung war nicht vollständig vor Ergebnissicht von S festgelegt.

Nachtrag 1 erlaubt die exakten Einzelfehler aus Lauf A155-2026-10-04-d6e70e, `publication_preparation/research-beat-boundaries-20261012-v1/errors-*.parquet`. Schema/Footer zuerst; nur `scope`, `source`, `landmark`, `error_ms`, Filter `scope=fused_global`, `source=fused`, gespeicherte Landmarks `r_onset`, `r_offset`, `t_offset`, endliche Fehler. `r_onset/r_offset` entsprechen QRS-Beginn/-Ende (Auftrag nennt qrs_onset/qrs_offset); Release `core/delineation/base.py` bindet das QRS-Segment an dieses Paar. Nenner 1831/1831/1642; SD 13,522707/11,593511/32,795118 ms, Wertebereiche −70 bis 80 / −58 bis 154 / −160 bis 332 ms. Alle Histogrammklassen exakt reproduziert, einschließlich der T-Ende-Fehler über 200 ms. Kein Ausreißer entfernt oder abgeschnitten. Quellenhashes sämtlicher 200 Dateien in `E_errors_audit.json`.

Seed 20261012; `numpy.default_rng`, EKGs numerisch nach study_id, je EKG 20 Züge, je Zug Grenzen in Reihenfolge QRS-Beginn/QRS-Ende/T-Ende; je Grenze unabhängige gleichverteilte Auswahl mit Zurücklegen, Vorzeichen umgekehrt, `round(-error_ms/2)` Samples bei 500 Hz. Alle gezogenen Fehler liegen bereits auf dem 2-ms-Raster. Private Ziehungen/Checkpointwerte ausschließlich unter `<local path>`.

## Szenario E: Kennzahlen und Deutung nach §6

SD_stör ist die Stichproben-SD (ddof=1) der **vorzeichenbehafteten Differenzen gestört minus ungestört**, gepoolt über EKGs und Züge, nur endliche Paare. Sie ist weder RMSE noch mittlere absolute Änderung und umfasst keine nicht berechenbaren Fälle. R = SD_stör / within-subject SD aus A, Datei `{repeat}`, population=gesamt, exakte marker_id. Nenner ist `sem=difference_sd/√2` aus `werkzeuge/rv_a1/wiederholbarkeit.py`; klinische Wiederholungsstreuung einschließlich möglicher Zustandsänderung, keine isolierte Messfehlervarianz. Nennerunsicherheit nicht neu geschätzt.

ρ = Median der 20 zugweisen Spearman-Korrelationen zwischen Baseline und gestörtem Wert, je Zug endliche Paare. Kein Poolen der 10.000 Werte als unabhängige Rangpaare und keine Korrelation mit dem Median der gestörten Werte. 95-%-Perzentilintervalle aus 1.000 Ziehungen über 500 EKG-/Personencluster (Seed 20261012), sämtliche 20 Züge eines gezogenen EKGs gemeinsam; SD und alle 20 Korrelationen jeweils neu berechnet, gemeinsame Bootstrapindizes aller Marker. Unsicherheit bedingt auf LUDB-Fehlerquelle, train-Auswahl und festen A-Nenner; kein Bootstrap der LUDB-Fehler oder Wiederholungspaare.

{chr(10).join(table)}

Kategorien ausschließlich nach Punktschätzer und §6: R<0,5 grenzrobust relativ zur Kurzzeitstreuung; 0,5≤R<1 mäßig grenzempfindlich; R≥1 grenzempfindlich. Ein Intervall über einer Kategoriegrenze bedeutet Unsicherheit der Einordnung, keine neue Erfolgsregel. {robust_sentence}

### Nichtberechenbarkeit und effektive Nenner

| Marker | EKGs mit mindestens einem Fehlwert | EKGs ohne berechenbaren Zug | endliche Paare je Zug (Min-Max) | berechenbare ρ-Züge / 20 |
| --- | --- | --- | --- | --- |
{missing_table}

Kein Ersatz-EKG, keine Imputation; Fehlwerte bleiben Abstentionen. Die ausgewiesene SD/ρ beschreibt nur berechenbare Störungen. Informative Ausfälle können Robustheit günstiger erscheinen lassen; Missingness ist eine eigene Grenze.

### Fréchet gegen Winkel

Primärmarker R={fmt(primary['R'])}, 95-%-KI {ci(primary,'R')}; Winkel R={fmt(angle['R'])}, 95-%-KI {ci(angle,'R')}. {comparison} Dies ist der spezifizierte Intervallvergleich, kein zusätzlicher Signifikanztest. Keine Änderung einer B-Testzahl.

## Szenario S: mittlere absolute Änderung
{s_section}
S wurde nicht neu berechnet; sämtliche Messwerte, Intervalle und Fehlwertzahlen der 36 S-CSV-Zeilen wurden identisch übernommen. Nur das historische Kategorie-Metadatum „R aus E fehlt“ wurde durch „nicht anwendbar; Kategorie nur aus E“ ersetzt.

## Claimgrenzen und F1-F8

| Falle | Stellungnahme für jeden der sechs Marker |
| --- | --- |
| F1 | nicht_zutreffend: kein Endpunkt geladen; technische mechanische Grenzabhängigkeit ist Prüfgegenstand, kein biologischer Claim |
| F2 | eingeschraenkt: nur train, gemeinsame Endlichkeitsselektion, Krankenhauspopulation; keine Prävalenz-/PPV-/Transportaussage |
| F3 | nicht_zutreffend: keine diagnostische Schwelle gewählt; feste §6-Grenzen unverändert |
| F4 | eingeschraenkt: alle sechs Marker berichtet; Quelle nach Sicht auf S präzisiert und vor E festgelegt, Zeitpunkt offengelegt; keine Gewinnerauswahl |
| F5 | offen: gemeinsame Endlichkeitsselektion und markerweise E-Abstentionen; keine MAR-Annahme oder Robustheit aus Ausfällen |
| F6 | eingeschraenkt/offen: spezifizierter Primärmarker-Winkel-Vergleich oben; keine allgemeine Gleichheit der Messgüte oder Inkrementfreigabe |
| F7 | eingeschraenkt: ein EKG je Person und Clusterbootstrap; A-Nenner andere Population/Zustandsfreiheit nicht belegt |
| F8 | offen: LUDB andere Population/Entwicklungsnutzung, unabhängige Grenzstörung ignoriert Fehlerkovarianz; weitere Geräte-/Filter-/Pacingreste ungeprüft |

P01a: Hashbindung beider Spezifikationen und Checkpoints. P01b: alle Histogrammklassen exakt; Aggregat-SD unabhängig gegen numpy.std und Rangverfahren gegen scipy.stats.spearmanr geprüft (siehe Abschlussverifikation). P01c: vollständige ungestörte Äquivalenz 500/500; keine unabhängige manuelle EKG-Adjudikation. P03: 500 Personen, 500 EKGs und 10.000 Störfälle getrennt, endliche Nenner markerweise. P08: Personencluster, gemeinsames Resampling, Perzentil-KI und bedingtes Unsicherheitsziel oben. Keine externe Bestätigung/klinische Freigabe.

## Auflagen, Widerlegung und offene Punkte

MS: §6-Deutung und Missingnessbegrenzung wie oben; PR: unveränderter Release und vollständige Äquivalenz. Kein Dissens. K-Abnahme getrennt von Auftragserledigung. Unbelegte Transport-/Inkrementclaims bleiben offen. Eine gescheiterte Checkpointbindung, Äquivalenz, Histogrammreproduktion oder unabhängige Aggregatprüfung würde die technische Aussage ungültig machen; diese Prüfungen sind in der Abschlussverifikation dokumentiert. Zusätzliche klinische Grenzfehlerpopulationen und Messgütereste sind außerhalb dieses Auftrags, nicht als erledigt behauptet. Keine noch erforderliche Rechnung aus AW143 offen.

## Quellenbindung

Rezept E: `empirical.py::errors/run/shifted_all` → private Checkpoints → `empirical_report.py::report/rho_draws/verify`. S: ursprüngliches `run.py`/`report.py`, unverändert. Ursprünglicher Stopbericht und CSV als historische Aggregate auf E: erhalten. Historische Quellenhashes:

| Quelle | SHA-256 |{old_hash_table}

Zusätzliche Bindungen:

| Quelle | SHA-256 |
| --- | --- |
{hashes}

Abschlussverifikation und Ausgabehashes: `<local path>`. Speicherbelege `resources_start_E_*.json`, `resources_end_E_*.json`; ein Pythonprozess, Budget ≤1,5 GiB, keine Installation oder fremde Prozessbeendigung. Keine Literaturrecherche; keine Quellenfehler-Kandidaten fremder Literatur.

Sicherheit: hoch für technische Reproduktion und bedingte §6-Klassifikation; begrenzt für klinische Messgüte, Missingness und Transport. Kein Manuskript erstellt, keine Manuskriptplatzhalter.
'''
    target.write_text(text,encoding='utf8')
    abgabe=f'''# Abgabe AW-143 - B-Störtest

## Ergebnis

**Auftrag vollständig gerechnet: S unverändert übernommen, E mit 500 train-EKGs / 500 Personen × 20 Zügen und 1.000 Clusterbootstrapziehungen abgeschlossen.**

{cats}.

{comparison}

Beide Spezifikationshashes stimmen; vollständige ungestörte Äquivalenz und erneute E-Äquivalenz 500/500 bestanden (rtol=1e-8, atol=0). Alle Histogrammklassen aus den exakten Fehlern reproduziert; keine Ausreißer entfernt. Vollständige Werte, Intervalle und Nichtberechenbarkeit im Bericht/CSV.

Nachtrag laut Auftrag 12.10.2026 etwa 21:00 **nach Sicht auf S, vor E**; offen im Bericht dokumentiert. Auftrag erledigt; K-Abnahme und klinische Claimfreigabe separat.

## Geänderte/neue Dateien

- `nachrechnung/B_STOERTEST_AW143.md`, `nachrechnung/b_stoertest.csv` - Ergänzung E, Kategorien und Vergleich; S-Messwerte/Intervalle/Fehlwertzahlen unverändert, historische S-Kategoriemetadaten aktualisiert.
- `werkzeuge/b_stoertest/empirical.py`, `empirical_report.py`, `start_e.ps1` - Fortsetzung, Checkpoints, Aggregation und Verifikation.
- `<local path>` - private Ziehungen/E-Checkpoints, Quellen-/Speicher-/Abschlussbelege und historische Aggregate.
- `koordination/abgaben/AW-143.md` - diese Abgabe.

## Belege und Reproduktion

`Get-FileHash nachrechnung/SPEZIFIKATION_B_STOERTEST_v1.md,nachrechnung/SPEZIFIKATION_B_STOERTEST_v1_NACHTRAG1.md -Algorithm SHA256`: `{SPEC_HASH}` / `{ADD_HASH}`.

`werkzeuge/b_stoertest/start_e.ps1 -Phase inspect`; `... -Phase run -Limit 1` (Pilot); `... -Phase run` (Checkpointfortsetzung); `... -Phase report`; `... -Phase verify`. Ausführung mit der Auswertungs-Pythonumgebung und `-B`.

Ein Prozess zugleich, vor jedem Start RAM/Commit ≥1,5 GiB über GlobalMemoryStatusEx, Budgetüberwachung. Speicher- und Reproduktionsnachweise in `E_abschlusspruefung.json` und `resources_*_E_*.json`. Keine gebundenen Werkzeuge oder Releasequellen verändert, kein fremder Prozess beendet, keine Installation. S-Checkpoints unverändert; kein erneuter S-Lauf. Personenbezogenes ausschließlich auf E:.

Quellen E: A155-2026-10-04-d6e70e, `research-beat-boundaries-20261012-v1/errors-*.parquet`; nur scope/source/landmark/error_ms und freigegebene fused_global/fused-Filter. Gespeicherte r_onset/r_offset entsprechen qrs_onset/qrs_offset im Auftrag; Nenner 1831/1831/1642. Alle 200 Quellenhashes in `E_errors_audit.json`. Keine B-Endpunkte, Labels, klinische Referenzen oder Testpartitionen geladen.

SD_stör: gepoolte signierte Änderungen, ddof=1; R-Nenner aus `rv_a1/repeat_marker_metrics.csv`, population=gesamt/marker_id, sem; ρ Median der 20 zugweisen Rangkorrelationen. 1.000 Personenbootstrapziehungen, alle Züge gemeinsam; fixe LUDB-Quelle und A-Nenner, keine Nennerunsicherheit. 42 Ausgabezeilen (36 S + 6 E).

## Offene Fragen und Auflagen

Keine noch erforderliche AW143-Rechnung offen. K-Abnahme ausstehend. F1-F8 im Bericht; informative Abstentionen, Endlichkeitsselektion, LUDB-Populationsunterschied und unabhängige Grenzstörung bleiben Grenzen. R bezieht sich auf klinische Kurzzeitstreuung, nicht isolierte Messfehlervarianz. Kein Inkrement-/Transport-/klinischer Claim freigegeben.

## Nebenbefunde / Quellenfehler-Kandidaten

Historische Histogramm-Unbestimmtheit durch Nachtrag 1 aufgelöst. S-Nebenbefund einer QRS-Daueränderung bei alleiniger T-Ende-Störung bleibt dokumentiert: mögliche Release-Schlag-/Repräsentativauswahl, technische Schlussfolgerung ohne schlagweise Adjudikation. Keine Fehler fremder Literaturquellen behauptet. Kein Manuskript, keine Platzhalter.

## Sicherheit

Hoch für technische Herkunft, Reproduktion und bedingte §6-Einordnung. Begrenzt für klinische Messgüte und Transport; keine klinische Freigabe. Ergebnisbericht und CSV bilden alle beauftragten Kennzahlen ab.
'''
    (PROJECT/'koordination/abgaben/AW-143.md').write_text(abgabe,encoding='utf8')

def verify():
    import numpy as np
    from scipy.stats import spearmanr
    baseline,values=load();sd,rho,dr,pn=estimates(baseline,values)
    err=errors()
    assert all(np.equal(v/2,np.rint(v/2)).all() for v in err.values())
    from werkzeuge.b_stoertest.run import readp
    cohort=readp(OUT/'selection_private.parquet',['subject_id','study_id','split'])
    assert len(cohort)==500 and cohort.subject_id.nunique()==500 and cohort.split.eq('train').all()
    assert {str(int(x)) for x in cohort.study_id}=={p.stem for p in (OUT/'records_E').glob('*.json')}
    audit=json.loads((OUT/'E_errors_audit.json').read_text(encoding='utf8'))
    for p,info in audit['sources'].items():assert sha(p)==info['sha256']
    with (PROJECT/'nachrechnung/b_stoertest.csv').open(encoding='utf8') as f:rows=list(csv.DictReader(f))
    with (OUT/'b_stoertest_vor_E.csv').open(encoding='utf8') as f:old_s=[r for r in csv.DictReader(f) if r['scenario']=='S']
    new_s=[r for r in rows if r['scenario']=='S'];e=[r for r in rows if r['scenario']=='E']
    assert len(rows)==42 and len(e)==6 and len(new_s)==36
    assert [{k:r[k] for k in o if k!='category'} for r,o in zip(new_s,old_s)]==[{k:v for k,v in o.items() if k!='category'} for o in old_s]
    assert all(r['category']=='nicht anwendbar; Kategorie nur aus E' for r in new_s)
    for j,r in enumerate(e):
        assert r['marker_id']==IDS[j] and int(r['n_computable'])+int(r['n_not_computable'])==10000
        delta=values[:,:,j]-baseline[:,None,j];finite=delta[np.isfinite(delta)]
        assert np.isclose(float(r['sd_stoer']),np.std(finite,ddof=1),rtol=1e-12,atol=0)
        independent=[]
        for draw in range(20):
            mask=np.isfinite(values[:,draw,j])
            correlation=spearmanr(baseline[mask,j],values[mask,draw,j]).statistic
            assert np.isclose(correlation,dr[draw,j],rtol=1e-12,atol=1e-14)
            independent.append(correlation)
        assert np.isclose(float(r['rho']),np.median(independent),rtol=1e-12,atol=1e-14)
        assert np.isclose(float(r['R']),float(r['sd_stoer'])/float(r['within_subject_sd']),rtol=1e-12)
        for key in ['sd','R','rho']:assert float(r[key+'_ci_low'])<=float(r[key+'_ci_high'])
        ratio=float(r['R']);cat='grenzrobust relativ zur Kurzzeitstreuung' if ratio<.5 else 'mäßig grenzempfindlich' if ratio<1 else 'grenzempfindlich'
        assert r['category']==cat
    # Bootstrap-Ränge mit duplizierten Personen und Missingness unabhängig prüfen.
    ix=np.random.default_rng(SEED).integers(0,500,500)
    br,bn=rho_draws(baseline[ix],values[ix])
    delta=values-baseline[:,None,:]
    for j in range(6):
        d=delta[ix,:,j];finite=d[np.isfinite(d)]
        n=np.isfinite(delta[:,:,j]).sum(axis=1)[ix].sum()
        s=np.nansum(delta[:,:,j],axis=1)[ix].sum()
        q=np.nansum(delta[:,:,j]**2,axis=1)[ix].sum()
        assert np.isclose(np.sqrt((q-s*s/n)/(n-1)),np.std(finite,ddof=1),rtol=1e-12,atol=0)
    for j in range(6):
        for d in [0,9,19]:
            mask=np.isfinite(values[ix,d,j])
            expected=spearmanr(baseline[ix,j][mask],values[ix,d,j][mask]).statistic
            assert np.isclose(br[d,j],expected,rtol=1e-12,atol=1e-14)
    original=json.loads((OUT/'abschluss.json').read_text(encoding='utf8'))
    fixed={p:v for p,v in original['hashes'].items() if not p.endswith(('SPEZIFIKATION_B_STOERTEST_v1.md','repeat_marker_metrics.csv'))}
    for p,v in fixed.items():assert sha(p)==v
    s_hashes={p.name:sha(p) for p in (OUT/'records').glob('*.json')}
    outputs={str(p):sha(p) for p in [PROJECT/'nachrechnung/B_STOERTEST_AW143.md',PROJECT/'nachrechnung/b_stoertest.csv',PROJECT/'koordination/abgaben/AW-143.md']}
    save('E_abschlusspruefung.json',{'status':'bestanden','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'ecgs':500,'persons':500,'draws':10000,'bootstrap':1000,'output_rows':42,'S_all_fields_except_category_unchanged':True,'S_category_metadata_updated':True,
         'all_120_Spearman_draw_correlations_independently_checked':True,'bootstrap_duplicate_rank_checks':18,'all_source_errors_on_2ms_grid':True,
         'SD_numpy_std_independent':True,'fixed_sources_unchanged':fixed,'S_checkpoint_hashes_private':s_hashes,'outputs':outputs})
    print(json.dumps({'verification':'bestanden','rows':42,'E_results':[{'marker_id':r['marker_id'],'R':r['R'],'category':r['category']} for r in e]},ensure_ascii=False),flush=True)
