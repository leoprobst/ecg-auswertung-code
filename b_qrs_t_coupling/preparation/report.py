"""Deutschsprachiger Aggregatbericht und additive CSV; keine Patientenausgaben im Projekt."""
import csv,json,hashlib
from pathlib import Path
from .runner import PROJECT,sha,save

def load(out,name):return json.loads((out/name).read_text(encoding='utf8'))
def fmt(x):return 'nicht schätzbar' if x is None else f'{x:.6f}'.replace('.',',')
def ci(z,k):
    v=z.get(k+'_ci95')
    return 'offen' if v is None else '['+'; '.join(fmt(x) for x in v)+']'

def table(items):
    text='| Auswertung | n train / positiv | n validation / positiv | OR je train-SD | Marker-AUC↑ | AUC B4 | AUC B4+Marker | ΔAUC [95%-KI] | Richtung |\n|---|---|---|---|---|---|---|---|---|\n'
    for name,z in items.items():
        c=z.get('counts',{});tr=c.get('train',{});va=c.get('validation',{})
        raw=z.get('raw_auc_train_direction');ascending=raw if z.get('direction')==1 or raw is None else 1-raw
        text+=f"| {name} | {tr.get('n','-')} / {tr.get('positive','-')} | {va.get('n','-')} / {va.get('positive','-')} | {fmt(z.get('or_per_train_sd'))} {ci(z,'or_per_train_sd')} | {fmt(ascending)} | {fmt(z.get('baseline_auc'))} | {fmt(z.get('augmented_auc'))} | {fmt(z.get('delta_auc'))} {ci(z,'delta_auc')} | {z.get('direction','-')} |\n"
    return text

def rows(out,run):
    groups=[('verification_result.json',{'Verifikation':load(out,'verification_result.json')}),('clinical_results.json',load(out,'clinical_results.json')),
      ('ipw_result.json',{k:v for k,v in load(out,'ipw_result.json').items() if isinstance(v,dict)}),('signals_result.json',{k:v for k,v in load(out,'signals_result.json').items() if not k.startswith('_')})]
    output=[]
    for file,items in groups:
        for name,z in items.items():
            c=z.get('counts',{});tr=c.get('train',{});va=c.get('validation',{})
            output.append({'id':'AW121','scenario':name,'model':'B4','feature_id':name,'status':z.get('status','offen'),'n_train':tr.get('n',''),'positive_train':tr.get('positive',''),'n_validation':va.get('n',''),'positive_validation':va.get('positive',''),
              'or':z.get('or_per_train_sd',''),'beta':z.get('beta_per_train_sd',''),'development_sd':z.get('train_sd',''),'iterations':z.get('replicates',''),'source_file':str(out/file),'source_sha256':sha(out/file),'script':'werkzeuge/b_vorbereitung/runner.py',
              'claim':'Entwicklungsprobelauf; keine unabhängige Bestätigung','AW121_run_id':run,'AW121_delta_auc_validation':z.get('delta_auc',''),'AW121_delta_auc_ci95':json.dumps(z.get('delta_auc_ci95')),
              'AW121_baseline_auc_validation':z.get('baseline_auc',''),'AW121_augmented_auc_validation':z.get('augmented_auc',''),'AW121_direction':z.get('direction',''),'AW121_result_json':json.dumps(z,ensure_ascii=False,separators=(',',':'))})
    lv=load(out,'lvh_result.json');output.append({'id':'AW121','scenario':'LVH_LSB_AW105','model':'B1+Cj+Peak','status':lv['status'],'reason':'; '.join(lv['reason']),'source_file':str(out/'lvh_result.json'),'source_sha256':sha(out/'lvh_result.json'),'script':'werkzeuge/b_vorbereitung/lvh.py','claim':'Machbarkeit; kein Nullbefund','AW121_run_id':run,'AW121_result_json':json.dumps(lv,ensure_ascii=False)})
    return output

def run(out,run):
    v=load(out,'verification_result.json');c=load(out,'clinical_results.json');ip=load(out,'ipw_result.json');lv=load(out,'lvh_result.json');sig=load(out,'signals_result.json');syn=load(out,'synthetic.json');audit=load(out,'audit.json')
    text=f'''# AW-121 - Vorbereitung der Sekundäranalysen von Hauptarbeit B

11.10.2026 · PR/MS · **Entwicklungsprobelauf, keine Testaussage und kein Studien-Freeze.** Laufverzeichnis `{run}`. Spaltenprojizierte Entwicklungsdaten und attestierte train/validation-Referenzen; ein Index-EKG je Person und Endpunkt. Alle AUC werden von train-Fits in validation beurteilt. Konfidenzintervalle sind punktweise Patientenbootstrap-Intervalle, keine unabhängige Bestätigung und keine Holm-Familie. Kein klinischer Entscheidungsclaim.

## 1. Verifikationsselektion (§9.6)

Index: erstes technisch auswertbares Erwachsenen-EKG nach Zeit und anschließend study_id, unabhängig von Echo-/Outcome-/Markerwert. Der Indikator folgt `labels.py:_echo_rules`: `COALESCE(has_tte_30d,FALSE)`. Null ist ausschließlich bei fehlendem TTE-Ereignisjoin als kein Echo gewertet; Echo-Befunde und unbekannte Diagnosen gehen nicht in das Verifikationsmodell ein. ±7 Tage ist als zweiter Indikator/Fallzahlfluss geführt. Alle Index-EKGs: **{v['all_index_n']}**; vollständige B4-/Markerbasis: **{v['complete_n']}**. Vollständige Foldnenner:

```json
{json.dumps(v['all_index_by_fold'],ensure_ascii=False,indent=2)}
```

{table({'Echo im ±30-Tage-Fenster':v})}

Komparator Alter/Geschlecht/QRS: validation-AUC {fmt(v['comparator_age_sex_qrs_auc'])}. B4 und B4+Marker werden auf derselben vollständigen Basis verglichen. Das B4-Gewichtsmodell enthält **keinen Kopplungsmarker**. P99 aus verifizierten train-Personen: {fmt(v['p99_cap'])}; Propensity-Bereich {v['propensity_range']}; maximale ungekappte Gewichte {fmt(v['weight_max_before'])}. Modellhash vor den hier berichteten LVEF-Fits: `{v['weight_model_sha256']}`. Gespeicherte Parameter/P99 und aktuelle Default-Implementierung sind unabhängig reproduziert (`audit.json`); die technische Korrekturfolge steht in §8.

Belastbarkeit: Der Verifikationsfit ist ein interner Complete-case-Befund; er beschreibt beobachtete Echo-Selektion und räumt ungemessene Indikations- und Missingnessmechanismen nicht aus.

## 2. LVEF-Kontext und Gegenkontrollen (§9.7a/b)

Primäre Referenz bleibt `<40 %`; die in §9.6/9.7 genannte inklusive `≤40 %`-Definition steht getrennt daneben. Hypertrophie-Kontrolle: LVEF≤40 gegen positive massenbasierte `echo_lvh_strict` mit LVEF≥50, dasselbe Echo je Index. RV-Kontrolle: LVEF≥50 und attestierte binäre eligible RV-Referenz am RV-Index. Haupt-RV-Marker DTW gemäß §1; Fréchet daneben deskriptiv. B4 enthält den jeweiligen registrierten B2-Komparator (LVEF: QRS-Dauer, RV: `classical_dominant_r_v1`).

{table(c)}

Echo-ID, Zeitabstand und Timestamp-Referenz werden pro Zeile auf Gleichheit geprüft. Unknown/unavailable/indeterminate bleiben ausgeschlossen, niemals negativ. Fallfluss: `clinical_flow.json`; inklusive exakt40-Grenzfälle. Die aktuellen Quellen markieren die konfirmatorische Zeitzuordnung durchgehend als nicht geeignet; daher ist dies eine begrenzte Entwicklungsrechnung auf attestierten eligible Timestamp-Referenzen.

Belastbarkeit: Diese Kontrollen prüfen Spezifität innerhalb der verifizierten Entwicklungspopulation; weder ein positives Inkrement noch ein Intervall mit null beweist einen eigenen Krankheitsmechanismus bzw. Gleichwertigkeit.

## 3. Verifikationsgewichtung (§9.6)

Kohärentes Fallset: erster technischer Index mit danach verifizierter LVEF≤40. Es wird mit ungewichteter Auswertung **desselben** Indexsets verglichen; das erste verifizierte EKG ist ein anderer Auswahlprozess und erhält nicht still dieselben Gewichte.

{table({k:z for k,z in ip.items() if isinstance(z,dict)})}

In jedem der 1.000 Replikate: gemeinsame Patientenmultiplizitäten im Verifikations-/LVEF-Set, neuer train-Verifikationsfit, train-P99, vollständige train-LVEF-Neufits und gewichtete gepaarte validation-AUC. Die Original-OR-SD bleibt train-fix. Vorzeichenumkehr: **{ip['direction_reversed']}**. Effektives gewichtetes N: {fmt(ip['IPW'].get('effective_n'))}. Dies ist keine Übertragung auf ungesehene oder nicht verifizierte Fälle.

Belastbarkeit: IPW setzt ignorierbare Selektion nach beobachteten Kovariaten und ausreichende Positivität voraus; Complete-case-Selektion und ungemessene Echo-Indikation bleiben offen.

## 4. Matrixfreie Kopplung und inverse Dower (§9.5/9.3)

Deterministischer Signalpilot: höchstens 200 Personen je Fold aus der bereits vorhandenen, nach Outcome/QRS/Rhythmus geschichteten AW023b-Stichprobe, Identitätshash Seed121 ohne neue Auswahl nach Marker oder Ergebnis. Alle Transform-/Zerlegungsfits verwenden ein gemeinsames vollständiges Fallset. Rohsignale bleiben auf E:. Dieselbe Medianbildung/Delineation wird für Kors, inverse Dower und die acht unabhängigen Ableitungen I, II, V1-V6 benutzt. Originalreproduktion rtol1e−8/atol0; ungültige Fälle abstinieren.

{table({k:z for k,z in sig.items() if k in ('original','dower','direct8_centered_rms','direct8_qrs_max_centered','kors_qrs_max_centered')})}

Fluss/Abstentionen: `{json.dumps(sig.get('_flow',sig),ensure_ascii=False)}`.

**Definitionsabweichung:** Der tatsächliche Release zentriert beide Schleifen getrennt und normiert jede auf RMS. §9.5 nennt QRS-Maximalnorm. `direct8_centered_rms` überträgt den ausführbaren Release-Vertrag; `direct8_qrs_max_centered` und `kors_qrs_max_centered` sind gesonderte technische Varianten mit gemeinsamem QRS-Maximum nach Zentrierung. Sie werden nicht als identische Primärmarker oder als abgenommene normative Auslegung ausgegeben. Ein unqualifizierter Transformationsrobustheitsclaim bleibt deshalb gesperrt.

Belastbarkeit: Der Signalpilot prüft Rechenweg und Richtung in einem begrenzten stratifizierten Entwicklungsset; er ersetzt keine Vollkohortenrechnung oder unabhängige Mess-/Herzfunktionsvalidierung.

## 5. Deskriptive Zerlegung (§9.2)

Original, train-fixes Residuum und rotationsbereinigter Formrest auf exakt dem gemeinsamen Signalset:

{table({k:z for k,z in sig.items() if k in ('original','residual','shape')})}

Formrest: VB-VF01-07 unverändert, 64 Bogenlängenpunkte, getrennte RMS-Normierung, orthogonales Procrustes mit Determinantenkorrektur. Residuum: VB-VF01-08 unverändert, Huber epsilon1,35/alpha0, QRS/QT/HF/Alter plus kategoriales Geschlecht, train-fixe Mittel/SD(ddof1). Parameter und Rezepthashes in `residual_fit_frozen.json`. Die knappe §9.2-Aufzählung nennt Geschlecht nicht, das ausdrücklich referenzierte Originalrezept schon. Erwartete Reihenfolge wird nicht durch Wahl einer günstigeren Komponente erzwungen. ΔAUC ist bedingter Zusatz über B4, nicht der unadjustierte AUC-Abstand der Komponenten.

Die Spalte Marker-AUC↑ verwendet höhere Originalwerte als höheren Score, ohne nachträgliche Ausrichtung nach validation. Die train-angepasste bedingte Koeffizientenrichtung wird separat ausgegeben. Die zugehörigen Marker-AUC-Intervalle stehen in den Ergebnisobjekten; bei negativer train-Richtung erhält man das Intervall der steigenden Originalwerte durch [1−oberer Grenze;1−unterer Grenze].

Belastbarkeit: Die Zerlegung beschreibt diesen begrenzten Entwicklungsversuch; eine kleine oder unsichere Formrest-AUC ist kein Beweis, dass jede Forminformation biologisch bedeutungslos ist.

## 6. LVH bei LSB (AW105)

Status: **{lv['status']}**; primäres ΔAUC **{fmt(lv.get('delta_auc'))}**, kein biologischer Nullbefund aus einem Vertragsstopp. Ausführungstore:

'''
    text+='\n'.join('- '+x for x in lv['reason'])+'\n\n'
    text+='| Fold | alle Indexpersonen | LSB+ | Echo im Fenster | massenverifiziert | gemeinsam C01-C08 | LVH+ | LVH− |\n|---|---|---|---|---|---|---|---|\n'
    for s,z in lv['counts'].items():text+=f"| {s} | {z['all_index']} | {z['LSB_positive']} | {z['LSB_echo']} | {z['LSB_mass_eligible']} | {z['joint_C01_C08_n']} | {z['positive']} | {z['negative']} |\n"
    if lv.get('primary_fit_executed'):
        text+=f"\nAttestierter Primärfit: train-Auswahl {lv['selected']}; validation-ΔAUC {fmt(lv['delta_auc'])} {ci(lv,'delta_auc')}; erfolgreiche Replikate {lv['successful']}/{lv['requested']}. Primärnenner: `{json.dumps(lv['counts_primary'])}`.\n"
    text+='''
Der implementierte Auswahlkern nutzt B1 inklusive linearer Alters-Hinge-Basis und QRS×Geschlecht, train-only fünf innere Patientenfolds, OOF-AUC-Auswahl und vollständige Bootstrap-Neuwahl. Kandidaten C01-C08 sind aus vorhandenen Größen formulierbar. C09/C10 werden wegen fehlender vollständig geprüfter Originalregeln nicht still weggelassen. Ohne datierten Spezifikationsnachtrag und Pacing-/WHO-Vertrag gibt es keinen zulässigen train-gewählten Hauptkomparator. Synthetischer Test prüft Train-only-Auswahl und Multiplizitätsbehandlung; Patientenauswahlkern bleibt absichtlich hinter dem Vertragstor.

Belastbarkeit: Die Fallzahlen belegen Machbarkeit unter den vorhandenen Surrogaten; sie liefern weder ein zulässiges AW105-ΔAUC noch eine klinische LSB-Diagnose.

## 7. Technische Prüfung und FE-20

Fréchet gegen unabhängig rekursiv gelösten Kleinstfall; Translation, gemeinsame Skalierung, Achsenpermutation, degenerierte/NaN-Kurven; QRS-Peak gegen höhere T-Spitze außerhalb QRS und Z-Spiegelung; Dezimalgrenzen40/50/95/115/120/60/720/168. Gewichtete AUC unabhängig gegen sklearn, frozen Fit/P99/Propensity unabhängig reproduziert, ΔAUC-Arithmetik und Fold-/Personendisjunktheit geprüft.

'''
    text+='| Dimension | Normierung | 500 Hz | 1.000 Hz | relative Änderung |\n|---|---|---|---|---|\n'
    for z in syn['FE20']:text+=f"| {z['dimensions']} | {z['normalization']} | {fmt(z['500_Hz'])} | {fmt(z['1000_Hz'])} | {fmt(z['relative_difference'])} |\n"
    text+='''
Belastbarkeit: Alle vier synthetischen Fälle liegen unter der vorab gesetzten technischen 1%-Toleranz; dies räumt FE-20 für reale Daten, Filter und Delineation nicht generell aus. Es gab keine realen Vorher-/Nachher-Paare, daher ist AE075/090 nicht anwendbar.

## 8. Abweichungen und offene Entscheidungen

1. Widerspruch QRS-Maximum (§9.5) gegen getrennte RMS-Normierung im Release. Beide technische Varianten getrennt; K/MS müssen den endgültigen normativen Vertrag klären.
2. Aktuelle konfirmatorische Zeitzuordnung nicht geeignet. Nur Entwicklungs-Pilot mit attestierten eligible Timestamp-Labels; keine Öffnungs-/E3-Freigabe.
3. AW105: C09/C10-, sicherer Pacing-/Präexzitations- und WHO-Vertrag fehlen. Kein Ersatzfit nach Sicht von validation, kein Haupt-ΔAUC unter verkürzter Kandidatenliste.
4. Signalrechnung auf festem Pilotset statt Vollkohorte. Quelle bereits nach Outcome geschichtet; kein Prävalenz-/Transportclaim. Die Pilotgröße bleibt für den unveränderten Wiederlauf fest.
5. IPW verwendet kohärent den ersten technischen Index; ungewichteter Primärkontext behält den ersten verifizierten Index. Die Resultate sind unterschiedliche deskriptive Estimanden und werden nicht vermischt.
6. Sparse TTE-Nulls waren beim ersten technischen Versuch zunächst als unbekannter Indikator behandelt. Die Diagnose zeigte ausschließlich fehlende TTE-ID und Zeit; korrigiert gemäß Release-COALESCE vor erstem LVEF-Fit und neu gehasht. Ein früher klinischer Gateversuch blieb ohne Fit, weil die konfirmatorische Zeitmarkierung für alle Fälle falsch ist. Keine Regeländerung nach LVEF-Ergebnissicht.
7. `<40 %` und `≤40 %` werden getrennt. Residualrezept enthält gemäß VB-VF01-08 auch Geschlecht. B4-Spline bleibt der ausführbare AW054-Vertrag; AW105 nutzt seine eigene explizite lineare Splinebasis.
8. Aus dem beauftragten Laufverzeichnis wird der manifestierte aktuelle Entwicklungscache verwendet. Dortige `development_provenance.json` nennt inzwischen den adoptierten Single-marker-Lauf; keine Behauptung einer unveränderten historischen Vor-Adoptionsmatrix. Footer- und Projektionshashes binden den tatsächlich gelesenen Stand; vollständige 16-GB-Quellhashes wurden nicht neu berechnet.
9. Ein begonnener technischer Versuch verwendete Pipeline-HF in B4. Dies wurde beim Abgleich mit dem ausführbaren AW054-Vertrag korrigiert auf zeitlich geeignete Geräte-HF. Der Versuch wurde während des ersten klinischen Bootstrap abgebrochen, ohne fertige klinische Ergebnisse; überholtes Gewichtsmodell/Verifikationsergebnis unter `superseded/` archiviert. Der korrigierte Gewichtsfit wurde vor den hier berichteten LVEF-Fits neu gebunden. Da bereits ein vorläufiger Punktfit begonnen hatte, ist dies kein makelloser Erstzugriff und keine rückwirkende Präregistrierung; die Korrektur folgt dem vorhandenen Vertrag, keiner Leistungsoptimierung. Pipeline-HF bleibt nur im ausdrücklich unveränderten Residualrezept.
10. OR werden im Probelauf je Standardabweichung des jeweiligen train-Fallsets beschrieben (in den Replikaten fest). Sie sind kein Ersatz für die später vor Testöffnung zu bindende gesamte Entwicklungs-Skalierung. SDs stehen in allen Ergebnisobjekten; OR verschiedener Kontrollpopulationen/Komponenten sind daher nicht unmittelbar vergleichbar. Der spätere konfirmatorische Vertrag bleibt ein gesondertes Freigabepaket.

## 9. Fallen F1-F8 und Fachstatus je Abschnitt

| Auswertung | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|---|---|---|---|---|---|---|---|---|
| Verifikation/IPW | eingeschränkt: Indikator ohne Echobefund | eingeschränkt: Krankenhaus-/Complete-case-Auswahl | nicht zutreffend: kein Arbeitspunkt | eingeschränkt: Entwicklung, keine konfirmatorische Auswahl | offen: MAR/MNAR und Kovariatenverluste | eingeschränkt: B4 plus einfacher Komparator | offen: ungemessene Indikation, Zeitattestation | eingeschränkt: technische Index-/QC-Selektion |
| LVEF/LVH-Gegenkontrolle | eingeschränkt: gemeinsame Echoquelle, keine LVMI-Kovariate | eingeschränkt: verifizierte Extremgruppen | nicht zutreffend | eingeschränkt: deskriptiv, keine Holm-Familie | offen | eingeschränkt: B4, keine globale Optimalität | offen: Verifikation/Zeit | offen: Mess-/Delineationsrest |
| RV bei erhaltener LVEF | eingeschränkt: Echoquelle, konditionierte LVEF | eingeschränkt | nicht zutreffend | eingeschränkt: Entwicklung | offen | eingeschränkt: B4 mit RV-Komparator | offen | offen |
| Transformation/Zerlegung | eingeschränkt: Normalisierungsdefinition | eingeschränkt: stratifizierter Pilot | nicht zutreffend | eingeschränkt: feste Varianten, post-hoc Ursprung | offen: gemeinsames Kern-Fallset | eingeschränkt: gleiche B4-Basis | offen: Zeit/Verifikation | offen: synthetischer Test ersetzt reale Messgüte nicht |
| LVH bei LSB | offen: kein zulässiger Leistungsfit | eingeschränkt | nicht zutreffend | offen: Auswahlkern nur synthetisch | offen | offen: C09/C10 | offen | offen: Pacing/WHO/QC |

Keine Fallenbewertung als bestandenes klinisches Tor. Widerlegen/begrenzen würden Vorzeichenumkehr unter Gewichtung, Verlust des Zusatzes auf finaler gemeinsamer Basis, Transformationsabhängigkeit oder Abtast-/Delineationsartefakte. Nullnahe KI sind kein Äquivalenzbeweis. Unabhängige Prüfung, externe Kohorte und Messgüte-Gates bleiben offen.

## 10. Herkunft, Wiederlauf und Ressourcen

Erzeuger: `werkzeuge/b_vorbereitung/runner.py` mit Modulen clinical/models/kernels/lvh/signals/audit/report. Rezepte: `README.md`. Neustart mit nur geänderter Lauf-ID: `powershell -File werkzeuge/b_vorbereitung/start.ps1 -RunId A155-… -Phase all`. Entwicklungsattestation, genutzte Spalten/Filter, Footers, Projektionen und Erzeugerhashes: `feature_binding.json`, `clinical_binding.json`, `audit.json`; exakte Ergebnisquellen jeder Tabellenzeile in der additiven `nachrechnung/b_vorbereitung.csv` unter id=AW121 und AW121_run_id. Frühere AW054-Zeilen bleiben erhalten. Personenprojektionen ausschließlich E:, Projekt nur Aggregate/Code.

'''
    for p in sorted(out.glob('resources_*.json')):text+=f"- `{p.name}`: `{json.dumps(load(out,p.name),ensure_ascii=False)}`.\n"
    text+='\nSelbsteinschätzung: hohe Sicherheit für die dokumentierten technischen Prüfungen und arithmetischen Resultate; begrenzte wissenschaftliche Belastbarkeit wegen Entwicklungsauswahl, fehlender Zeit-/Selektionsattestation und der ausdrücklich offenen AW105-/Normalisierungstore. Quellenfehler-Kandidaten: kein neuer Literaturfehler; der oben belegte interne Spezifikations-/Release-Widerspruch ist an K zu entscheiden.\n'
    dest=PROJECT/'nachrechnung/B_VORBEREITUNG_AW121.md';dest.write_text(text,encoding='utf8')
    path=PROJECT/'nachrechnung/b_vorbereitung.csv';before=path.read_bytes();oldhash=hashlib.sha256(before).hexdigest()
    backup=out/'b_vorbereitung_vor_AW121.csv'
    if not backup.exists():backup.write_bytes(before)
    with path.open(encoding='utf-8-sig',newline='') as stream:reader=csv.DictReader(stream,delimiter=';');old=list(reader);fields=list(reader.fieldnames)
    new=rows(out,run);existing={(z.get('AW121_run_id'),z['scenario']):z for z in old if z['id']=='AW121'}
    add=[]
    for row in new:
        key=(run,row['scenario'])
        if key in existing:
            assert existing[key]['source_sha256']==row['source_sha256'],'CSV Ergebnisbindung geändert'
        else:add.append(row)
    for row in add:
        for key in row:
            if key not in fields:fields.append(key)
    if add:
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,delimiter=';');writer.writeheader();writer.writerows(old+add)
    save(out/'delivery.json',{'old_csv_sha256':oldhash,'old_rows_preserved':len(old),'new_rows':len(add),'report_sha256':sha(dest),'csv_sha256':sha(path),'run_id':run})
    print(json.dumps({'phase':'report','rows_added':len(add),'report':str(dest)}),flush=True)
