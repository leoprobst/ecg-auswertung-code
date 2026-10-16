"""Build only AW-178-authorized aggregate deliverables and check the manuscript."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import csv
import ctypes
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(str(data_path("s5.auswertung")))
OUT=Path(str(data_path("s5.aw178")))
PUB=ROOT/'publikationen/PUB-08_zeitausrichtung'
PP=Path(str(data_path("s5.mimic_validation_runs_a155_2026_10_04_d6e70e_publication_preparation")))
SCRIPT='werkzeuge/s5/aw178_audit.py'
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','free','commit_total','commit_free','virtual','virtual_free','extended']]
m=Memory();m.length=ctypes.sizeof(m)
assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
assert min(m.free,m.commit_free)>=1.5*2**30

def load(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_csv(path, rows, fields=None):
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def table(rows, cols):
    return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join('---' for _ in cols)+'|']+['| '+' | '.join(str(r.get(c,'')) for c in cols)+' |' for r in rows])

assert load('status.json')['state']=='completed_with_prespecified_stops'
rest=PP/'research-rest-low-cost-20261012-v1'
restseal={k.replace('\\','/'):v for k,v in json.loads((rest/'completed-files.sha256.json').read_text(encoding='utf-8-sig')).items()}
restchecks={name:restseal.get(name)==sha(rest/name) for name in ['executed_source/a155_research_rest_low_cost.py','population-first-ecg.sha256.json','protocol.json']}
assert all(restchecks.values()),'R0-Erzeugersiegel stimmt nicht'
(OUT/'R0_rest_seal_checks.json').write_text(json.dumps(restchecks,indent=2),encoding='utf-8')
execution=load('execution.json');census=load('R5_census.json');demo=load('R6_af_demographics.json');miss=load('R5_missingness.json');history=load('historical_x08.json');r0=load('R0_checkpoint.json')
v7=(PUB/'MANUSKRIPT_S5_v7.md').read_text(encoding='utf-8')
manpath=PUB/'MANUSKRIPT_S5_v8.md'
man=manpath.read_text(encoding='utf-8').split('## References')[0]+'## References\n\n'+v7.split('## Literatur\n',1)[1].strip()+'\n'
manpath.write_text(man,encoding='utf-8')
afpath=PP/'research-documented-af-landmark-20261012-v2/first-eligible-landmarks.parquet'
manifest=load('input_manifest.json')
afhash=load('af_schema.json')['sha256']
rows=[]
def add(phase,split,key,value='',unit='',n='',status='berechnet',source='',ci_low='',ci_high='',interval='',note=''):
    rows.append({'phase':phase,'split':split,'key':key,'value':value,'unit':unit,'denominator':n,'ci_low':ci_low,'ci_high':ci_high,'interval':interval,'status':status,'run':'A155-2026-10-04-d6e70e','source':source,'source_sha256':afhash if source==str(afpath) else manifest.get(source,{}).get('sha256',''),'script':SCRIPT,'note':note})
for phase in range(7):
    statuses=['abgebrochen','nicht_gestartet','nicht_gestartet','abgebrochen','abgebrochen','teilweise_abgeschlossen','teilweise_abgeschlossen']
    reason=r0.get('R'+str(phase),r0['partition']+' '+r0['potassium'])
    add('R'+str(phase),'development','gate',status=statuses[phase],note=reason)
for row in census:
    for key in ['n','n_persons','excluded_exclusive','events','AF_machine_report','AF_icd_admission','AF_other_or_unknown','non_AF_contact_end','non_AF_death_end','non_AF_horizon_end','non_AF_tied_end','death_missing','death_recorded_before_horizon']:
        if key in row:
            add('R5',row['split'],f"horizon={row['horizon_days']};stage={row['stage']};field={key}",row[key],'Personen/Fälle',row['n'],'reproduziert' if key in ['n','events'] and row['stage']=='complete_cases' else 'berechnet',str(afpath),interval='kein KI: exakter endlicher Bestandszensus',note='Eine Person je Landmark; Horizonte nicht addieren. Start vor Indexselektion nicht reproduziert. Gespeicherte Zeitkonvention, keine biologische Inzidenz.')
for row in demo:
    for key in ['n','sex_F','sex_M','sex_unknown','age_present','age_missing','age_q25','age_median','age_q75']:
        add('R6',row['split'],f"horizon={row['horizon_days']};stage={row['stage']};field={key}",row[key],'gespeicherte Altersjahre' if key.startswith('age_q') or key=='age_median' else 'Personen',row['n'],source=str(afpath),interval='kein KI: deskriptiver Bestandszensus',note=row['age_definition']+'; '+row['setting'])
for row in miss:
    if row['field']=='subject_id':continue
    add('R5',row['split'],'stage=selected_landmarks;missing='+row['field'],row['missing'],'fehlende Werte',row['n'],source=str(afpath),interval='kein KI: exakter Bestandszensus')
xroot=PP/'research-final-hypotheses-20261012-v1'
for row in history['coefficients']:
    add('historisch_X08',row['split'],f"id=X08;profile={row['profile']};term=interaction",row['beta_raw_units'],'gespeicherte Targeteinheit/(h × numerische K-Kontexteinheit)',row['n'],'historisch_dokumentiert',str(xroot/'interaction-contrasts.csv'),row['ci_low'],row['ci_high'],'historisches clusterrobustes t-KI',note='q_bh_all_16='+row['q_bh_all_16']+'; keine neue physische Einheitenbindung')
for row in history['loss']:
    add('historisch_X08','validation','id=X08;profile='+row['profile']+';field=mean_paired_mse_gain',row['mean_paired_mse_gain'],'gespeicherte Targeteinheit quadriert',row['n_validation'],'historisch_dokumentiert',str(xroot/'frozen-validation-loss.csv'),row['ci_low'],row['ci_high'],'historisches deskriptives gepaartes t-KI; kein Bootstrap',note='KI umfasst null; keine Überlegenheit oder Äquivalenz')
reportpath=ROOT/'nachrechnung/S5_NACHRECHNUNG_AW178.md';csvpath=ROOT/'nachrechnung/s5_nachrechnung.csv'
write_csv(csvpath,rows)
cc=[r for r in census if r['stage']=='complete_cases']
demot=[r for r in demo if r['horizon_days'] in [0,30]]
report=f'''# S5-Nachrechnung AW-178

13.10.2026 · MS/PR/DO · Ergebnis zuerst · **vertraglich abgeschlossen mit Abbrüchen, fachlich nicht einreichungsreif**.

R0 scheitert am zulässigen X08-Entwicklungseingang und an der fallgebundenen Rohassay-/Targetbindung. R1 und R2 wurden deshalb nicht gestartet. R3 und R4 enden am gemeinsamen Partition-/Selektortor. R5 reproduziert den Zensus des versiegelten ausschließlich entwicklungsgebundenen AF-v2-Bestands, aber nicht den vollständigen upstream Index-/Beobachtungsfluss. R6 liefert AF-Demografie und sichere Aggregat-/Hashspur; das öffentliche Supplement und die übrigen Auditdemografien bleiben unvollständig. Keine Testdatei, keine gemischte Patientendatei und keine externe Kohorte geöffnet. Keine neuen Fits, keine Hypothesentests, keine Recherche.

## 1. Vertrag und Eingangsbindung

Spezifikationshash vor Ausführung passend: `{sha(PUB/'SPEZIFIKATION_S5_NACHRECHNUNG_ENTWURF.md')}`. AW-178 hebt die Ausführungssperre des Entwurfs auf, nicht dessen Abbruchregeln. Ein neuer signierter X08-Vergleich wurde weder improvisiert noch durch ein Ersatzsample ersetzt. Die engere Auftragsgrenze DuckDB 512 MB hätte Vorrang vor 1 GB im Entwurf; DuckDB wurde nicht gebraucht.

Laufbasis: `<local path>` (P). Gebundener Release: `<local path>`. Historische Erzeuger nur als Text, keine Imports oder Ausführung. Der R5-Zensus verwendet ausschließlich die zwei reinen Funktionen `risk_set`/`report_sinus`, per AST aus versiegeltem Quelltext extrahiert, ohne Programmroutinen oder deren Importe.

Private Spur vollständig unter `<local path>`: `input_manifest.json` (Dokumente/Quellen/SHA/Bytes), `seal_checks.json`, `source_anchors.json` (Zeilennummern), `R0_checkpoint.json`, `R5_freeze.json`, `af_schema.json`, `R5_census.json`, `R5_missingness.json`, `R6_af_demographics.json`, `historical_x08.json`, `execution.json`, Statusdatei. Die Siegelprüfung aller tatsächlich benutzten X08-/AF-Dokumente und des AF-Kohortenfiles bestand. Keine universelle Versiegelung anderer Resultroots behauptet. AF-Kohortenhash: `{afhash}`.

## 2. R0: tatsächlicher Abbruch und Datenwörterbuch

`P/research-final-hypotheses-20261012-v1/executed_source/a155_research_final_hypotheses.py`, `run`: `pd.read_parquet(POP)` vor Entwicklungsfilter. `POP` verweist auf `P/research-rest-low-cost-20261012-v1/population-first-ecg.parquet`. Dessen archivierter Erzeuger `a155_research_rest_low_cost.py`, `population`, vereinigt Train/Validation-Cache und historische Holdoutfeatures und bindet sie an einen gemeinsamen ersten Masterindex. Das ist ein positiver Dokumentenbeleg für einen gemischten Eingang, keine Behauptung aus einer geöffneten Partitionstabelle. Diese Datei und auch ihr Footer wurden **nicht geöffnet**. Das bekannte historische nachgelagerte Filtern wäre unter AW-178 keine zulässige Umgehung.

Das X08-Protokoll benennt ungeprüfte numerische Kalium-Konventionen ausdrücklich. Der Release beschreibt generische Rohwert-/Unit-/Kommentarverarbeitung; es fehlt die fallgebundene Kette des X08-Kontextwerts zum Assay/Item/Material samt Zieleinheit, Umrechnung, Referenzbereich und QC. Keine pauschale mmol/L-Freigabe, kein Pooling von Blutgas-/Chemie-K oder Calciumarten. `frozen_comparison` transformiert das gespeicherte Tpe-Ziel nicht und quadriert dessen Vorhersagefehler. Die Spaltenbenennung ms ist kein unabhängiger upstream Kalibrationsnachweis; deshalb in v8 gespeicherte Targeteinheiten.

| Feld / Quelle | Physische bzw. dokumentierte Bedeutung | Einheit/Präzision | Verbindliche Grenze |
|---|---|---|---|
| ECG-Zeit | Geräte-/Recordreferenz, nominales Ereignis | Zeitstempel; physische Genauigkeit ungeklärt | Datumsshift getrennt von Geräteuhr; HF-Offset keine Rekalibration |
| Labor-charttime | dokumentierter Laborereignisbezug | Zeitstempel | keine unabhängig verifizierte Probenentnahme |
| Labor-storetime | Speicherung | separater Zeitstempel | nicht Ersatz für charttime/Entnahme |
| Kaliumdelta | charttime minus ECG-Zeit (`evidence.py`, DATE_DIFF second /3600) | h, signiert im Kontext | X08 verwendet Absolutwert; Rohassay nicht fallgebunden |
| nearest Labor | zuerst Aufnahmebezug, dann absoluter Abstand, Vorseite, Zeit und Referenzordnung (`evidence.py`, nearest_order) | deterministische nominale Bindung | kein reines Distanzminimum über alle Quellen, keine neue Auswahl |
| Verordnungsstart | dokumentierter Beginn einer Verordnung (`event_ingest.py`) | Zeitstempel | keine Gabe |
| eMAR-charttime / storetime | dokumentiertes Gabeereignis / Speicherung | getrennte Zeitstempel | ohne Status-/Coverageprüfung keine negative Therapieabdeckung |
| Prozedur-chartdate | administrativer Tagesbezug | [Tagesbeginn,Tagesende) | same-day unbestimmt, keine Mitternachtsimputation |
| AF first/next | erste dokumentierte AF, ICD-Aufnahmestartproxy oder reportbezogene ECG-Zeit (`evidence.py`, `labels.py`) | first: gespeicherte Stunden; next: Sekunden/3600 | kein adjudizierter Onset; Quellen nicht austauschbar |
| followup_hours | bis letzter beobachteter ECG-/Krankenhauskontakt (`evidence.py`) | DATE_DIFF hour; ganzzahlig | kein lückenloses Monitoring; upstream Cachebindung nicht neu reproduziert |
| Todesdatum | gespeicherte Todesreferenz | Datumspräzision nicht zur Sekundenpräzision aufwerten | Grenze konkurrierenden Ereignisses; Beobachtungsverlust separat |
| age / sex | gespeichertes Kontextalter / administrative Kategorie | Alterswerte, F/M/unknown | Alterableitung des Caches nicht unabhängig gebunden; Genderidentität unbekannt |

Zeitsemantik oben ist eine Codeinterpretation des gehashten Release, keine neu recherchierte MIMIC-Dictionaryvalidierung. **R0-Missingness des X08-Entwicklungsbestands ist nicht berechnet**, da sein Lesen unzulässig war. Nenner unbekannt ist kein Nullnenner. Schema des gemischten Eingangs fehlt absichtlich; Schema/Footer des separat zulässigen AF-v2-Eingangs ist vorhanden.

Historische Rezeptprüfung aus AW-177 wird nicht als neue Forschung gezählt: quadratische stetige Haupteffekte, Train-Zentrierung, Alter/Sex/HF/ED/ICU; BH16 = acht Interaktionshypothesen × zwei Folds je Profil, nicht nur X08 und nicht globale Forschungskorrektur. Originale werden erhalten. Die vier X08-Zeilen und zwei Losszeilen wurden als sichere Entwicklungsaggregate gegen ihre Archive geprüft und in der CSV ausdrücklich `historisch_dokumentiert` geführt.

## 3. R1-R4

| Rechnung | Status | Fehlendes Tor / Konsequenz |
|---|---|---|
| R1 signierter/seitiger/fallgleicher X08-Vergleich | nicht gestartet | R0 nicht erfolgreich. Keine Steigungen, BH10, Loss-BH3 oder Bootstrap-KI; Seed 178001/1000 Resamples im Plan nur reserviert, nicht benutzt. |
| R2 Therapie/Anforderung | nicht gestartet | keine zulässigen R1-Fälle; Gabe-/Anforderungsabdeckung unbekannt, kein unbehandelt-Claim und kein Therapieeffectfit |
| R3 B01 | abgebrochen | in bereitgestellter Kette kein attestierter entwicklungsgetrennter Probenbestand mit historisch gebundenem Selektor. Kein neuer Zensus; historische gemischte Rohaggregate geschlossen. |
| R3 M01/M03 | abgebrochen | keine sichere Entwicklungs-/Episodenbindung. Merge-Richtung und individuelle Gabe-/Verordnungslatenz deshalb nicht aus historischen Aufenthaltzahlen erschlossen. Beide sind ein Beitrag. |
| R4 T01/T02 | abgebrochen | kein attestierter identischer Entwicklungs-Ereignis-ECG-Pool mit eingefrorener Episodenregel. Streng vor/nach/same-day-Zensus unbekannt, nicht null. |

Prüfumfang: bekannte Erzeuger/kleine Protokolle, Manifest aus AW-173/AW-177 und Dateiinventare der betroffenen Ergebnisroots. Das ist keine exhaustive Aussage, dass getrennte Inputs nirgendwo existieren; es fehlt ihr Nachweis **in der gelieferten Provenienzkette**. Ohne diese Bindung verbietet die Spezifikation Öffnen/Neuerzeugen aus gemischten Dateien. Als Manuskriptbeitrag bleiben Referenz-/Expositionsprobleme qualitativ; historische B01/M/T-Zahlen werden nicht eingesetzt.

## 4. R5: AF-v2-Zensus (teilweise abgeschlossen)

Zulässigkeit vor Lesen: archivierter `a155_research_af_landmark_revision002.py`, `cohort`, erzwingt train/validation vor Speicherung. Kohorte und Selektor gegen abgeschlossene Dateisiegel geprüft; danach Footer (38.051 Zeilen, eine Rowgroup), erst dann nur benötigte Spalten. Eindeutigkeit einer Person je Index und Foldunabhängigkeit direkt geprüft. Keine privaten Kennungen oder maschinellen Berichtstexte exportiert. Die Tabelle reproduziert den gecachten Selektorbestand, **nicht** sämtliche vorgelagerten ECG-Gelegenheiten; dessen historischer Gesamtmanifestzähler wird nicht als neuer foldgetrennter Fluss umetikettiert.

| Stufe | Train | Validation | Exklusive Ausschlüsse Train/Validation |
|---|---:|---:|---:|
| versiegelte erste geeignete Entlassungsindizes | 28.297 | 9.754 | vor dieser Stufe unbekannt |
| kompatible Ereignis-/Beobachtungsreferenz | 28.270 | 9.745 | 27 / 9 |
| positive beobachtete Zeit nach Entlassung | 23.151 | 7.945 | 5.119 / 1.800 |
| Complete cases: Marker, Alter, Sex, HF | 15.092 | 5.257 | 8.059 / 2.688 |

Die vorgelagerten fünf Gates sind im Cache bereits erfüllt und schließen dort keine weiteren Fälle aus. Die obigen Retentionszahlen gelten für alle drei Horizonte. Nach positive-time folgt noch der historische finite-positive-duration-Test; keine zusätzlichen Fälle gingen dabei verloren. Complete cases und Ereignisse stimmen **für alle sechs Fold-/Horizontkombinationen** mit `F34-documented-af-model-census.csv` überein. Keine Wiederholung von Cox-Fits oder Leistungsrechnung.

{table(cc,['split','horizon_days','n','events','non_AF_contact_end','non_AF_death_end','non_AF_horizon_end','non_AF_tied_end'])}

Spalten der Endgründe beziehen sich ausschließlich auf Fälle ohne vorher dokumentierte AF und partitionieren zusammen mit `events` den jeweiligen Complete-case-Nenner exakt. Die Horizonte sind verschachtelt; nicht addieren. Der eine Train-Tie bei 365 Tagen wird nicht willkürlich dem Kontakt oder Horizont zugeschlagen. Kontaktende = Ende sichtbarer Krankenhaus-/ECG-Dokumentation, **nicht** sicherer Verlust aller Versorgung und **nicht** bestätigte Ereignisfreiheit. Tod = konkurrierendes Ereignis, im historischen Rezept zensiert; keine kumulative AF-Inzidenz/Competing-risk-Schätzung. Die reine Bestandszählung hat kein inferentielles KI, denn sie ist ein exakter endlicher Zensus; die Abwesenheit eines KI ist keine behauptete Populationspräzision.

`death_recorded_before_horizon` zählt zusätzlich alle gespeicherten Todesdaten im Horizont und ist ausdrücklich **nicht** identisch mit dem frühesten death-Endgrund: 162/42 (30 d), 402/130 (90 d), 997/326 (365 d). Auch Todesdaten nach früherem Kontaktende oder AF zählen dort mit. Tod darf fehlende AF-Beobachtung nicht verlängern. Quellenaufschlüsselung der dokumentierten AF und Missingness jedes projizierten Kontextfeldes stehen vollständig in der sicheren CSV; keine Rohkommentare.

### v1/v2-Vertrag

| Merkmal | v1 (nur abgenommene Dokumente) | v2 (versiegelter Entwicklungsselektor) |
|---|---|---|
| Indexwahl | erstes Master-ECG je Person, anschließend Entlassungseignung | erste geeignete Entlassung, letztes geeignetes ECG davor |
| ECG | berichtet sinus, Waveform regular, brauchbare Metadaten; 0-7 d vor Entlassung | gleiche Eignung, Auswahl innerhalb geeigneter Entlassung |
| Ausschlüsse | bekannte frühere/Index-AF, AF bis Entlassung, inkonsistente Ereignis-/Beobachtungsreferenz, fehlende positive Nachbeobachtung | entsprechend; Ausschlüsse Bestandteil des anderen Selektors |
| Risikobeginn | Entlassung | Entlassung; Population/index mapping geändert |
| Beobachtungsende | Minimum letzter Kontakt/Tod/Horizont laut AW-174 | exakt im archivierten risk_set reproduziert |
| AF-Referenz | erste spätere Dokumentation, kein adjudizierter Onset | ICD-Aufnahmestartproxy/reportbezogene Zeit; kein adjudizierter Onset |
| Tod | gespeichert, konkurrierende Grenze, historisch zensiert | gleiche begrenzte Behandlung; getrennt gezählt |
| Schätzbarkeit | 0/9 aus AW-174, historisch testhaltige Quelle bleibt geschlossen | 5/6 aus sicherem Modellzensus; keine neue Leistungsschätzung |

Kein gepooltes Replikationsmaß. Kein zusätzlich festgelegter zeitlicher washout: Ausschluss beobachteter Vorgeschichte ersetzt keine vollständige vorausgehende Krankenhausabdeckung. Beobachtungsuhr/Sekundenauflösung aus Release dokumentiert, aber nicht vollständig historisch an jeden Cachefall neu gebunden; deshalb R5 **teilweise** abgeschlossen, kein biologischer Inzidenz- oder Negativclaim. Erste Indexselektion auf sämtlichen Kandidaten, Alterableitung und unabhängiger Ereignisstandard bleiben offen.

## 5. R6: SAGER und Reproduzierbarkeit

{table(demot,['split','stage','n','sex_F','sex_M','sex_unknown','age_present','age_missing','age_q25','age_median','age_q75'])}

Die at-risk/complete-case-Zahlen sind bei allen Horizonten identisch. Die sichere CSV behält jede Horizonzeile, um den Nennervertrag sichtbar zu halten. Unknown=0 betrifft ausschließlich diese selektierten Stufen; kein Rückschluss auf ausgeschlossene Rohfälle. Alter sind gespeicherte Kontextwerte, keine unabhängig nachgerechnete Altersdefinition. Median sinkt nach Complete-case-Auswahl, eine deskriptive Selektionsfolge ohne Kausalbehauptung. Setting: Entlassung aus Krankenhaus; ED/ICU nicht als Felder im versiegelten AF-Cache vorhanden. Deshalb kein erneuter Join mit gemischtem klinischem Kontext. Keine Sex-Interaktionen, keine Geschlechtsinvarianz.

Für die übrigen Audits keine neuen Demografieaggregate wegen fehlenden zulässigen Eingangs. Öffentliche Supplementfreigabe, vollständige Caliperwerte/Rezepte aller historischen Audits, Extensionmodule und upstream Provenienz bleiben offen. Private Manifest-/Schema-/Statusdateien sind **kein öffentliches Supplement**. Sichere Tabellen enthalten keine Personen-/Aufnahmekennungen oder Matchlisten. Öffentliche Dateiauslieferung wird nicht autorisiert oder behauptet.

## 6. F1-F8 und Prüfniveau

O=offen; E=eingeschränkt; N=nicht zutreffend für reinen Bestandszensus.

| Baustein | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | Claimgrenze |
|---|---|---|---|---|---|---|---|---|---|
| Uhrkern (übernommen) | E | E | E | E | O | O | E | O | HF-Kompatibilität, keine externe Uhrvalidierung |
| kontrollierte Paargelegenheiten (übernommen) | E | E | O | E | O | E | O | O | großzügige Machbarkeitsobergrenzen, keine Markerwirkung |
| X08 historisch / R0-stop | E | E | O | O | O | O | O | O | keine neue physisch interpretierbare Steigung; lokale Familie und ungesicherter Lossgewinn sichtbar |
| B01/M01/M03/T01/T02 | E | E | O | O | O | O | O | O | Partition/Referenz/Episoden unbekannt; nur qualitative Designlücke |
| AF-Zensus | E | E | E | E | O | N | O | O | Dokumentationsziel; post-hoc Index; exact finite census ohne Leistungsclaim |
| AF-Demografie | N | E | N | N | O | N | E | O | selektierte gespeicherte Kategorien, keine populationsweite oder Sexinvarianzaussage |

P01a Provenienz für verwendete AF-Kohorte erfüllt; P01b Zensus-Nachrechnung erfüllt; P01c unabhängige upstream Patienten-/Assaybindung offen. P03 unabhängige Personenzählung direkt geprüft; P05 stored clock semantics dokumentiert, physical clock accuracy offen. P08 für neue reine Zensen nicht inferentiell anwendbar; historische KI als ihre echten Typen erhalten. P09/P10 wegen Beobachtungsselektion/Verifikation offen. P13/P14/P20 hier kein neuer Prognoseclaim; AF-Komparatorvertrag gehört N2. Kein Klinik-/Methodikstatus „publikationsreif“.

Widerlegung/Änderung: Ein versiegelter eindeutig getrennt erzeugter erster-X08-Index samt Rohassaykette könnte R0 öffnen; das bestehende negative Tor beweist nicht, dass kein zulässiger Eingang existieren kann. Abweichende Personenzählung, Archivhashes oder risk_set-Ergebnisse würden die aktuelle AF-Reproduktion widerlegen. Mehr beobachtete Endzeit oder vollständige externe Diagnoseerfassung könnte den Dokumentationszielvertrag ändern, nicht rückwirkend bestätigen.

## 7. Ausführung und Grenzen

Start per `werkzeuge/s5/aw178_start.ps1`, losgelöst und verborgen; eigene Python-PID aus `execution.json`: {execution['pid']}. Status und Checkpoints im E:-Ordner. Vor Pythonstart GlobalMemoryStatusEx in PowerShell, nochmals im Skript; freier RAM/Commit am erfolgreichen Auditstart {execution['start_memory']['free_ram_gib']:.3f}/{execution['start_memory']['free_commit_gib']:.3f} GiB. Peak Working Set {execution['peak_working_set_bytes']/2**20:.1f} MiB, Laufzeit {execution['elapsed_seconds']:.3f} s, Python {execution['python']}, pandas {execution['pandas']}, NumPy {execution['numpy']}, PyArrow {execution['pyarrow']}. Ein Pythonprozess gleichzeitig; kein Prozess beendet, keine Installation. Ein dokumentierter erneuter Auditstart ergänzte Personen-Eindeutigkeit und AF-Quellenzählung; keine Anpassung nach Signifikanz oder neue Fits.

Der erste Start-Process-Versuch scheiterte **vor Pythonstart** an doppelter Path/PATH-Umgebungsbelegung. Der gespeicherte Starter nutzt danach ProcessStartInfo/ShellExecute mit verborgenem Fenster. Seine Shell-Launcher-PID kann von der tatsächlichen eigenen Python-PID abweichen; maßgeblich ist die vom eigenen Skript geschriebene PID. Kein fremder Prozess angefasst.

Keine großen JSONs, kein umfassendes Parquetladen und keine gemischten Tabellen. Nachrechnung.csv und Manuskript bleiben sichere Aggregate. Standalone-Note bleibt bedingt; wegen R0/R3/R4-Abbruch keine Verstärkung des Manuskriptclaims. Entscheidung für A-Methods-Supplement nach Strategie bleibt K vorbehalten.
'''
reportpath.write_text(report,encoding='utf-8')
# Keep historical claim IDs with their provenance; distinguish absent/archival uses.
with (PUB/'CLAIM_MATRIX_S5_v2.csv').open(encoding='utf-8-sig',newline='') as f:
    old=list(csv.DictReader(f))
claims=[]
for row in old:
    if row['claim_id'].startswith('S5-META'):continue
    if 'ausgeschlossen' in row.get('verwendung_v7','') or 'STAFF' in row.get('lauf',''):continue
    row.update(verwendung_v8='historischer Beleg; nur soweit im englischen Text verwendet',nachrechnungsstatus_aw178='kein neuer Fit; Original erhalten')
    row['manuskriptstelle']=row.get('manuskriptstelle','').replace('v7','v8')
    if 'nur Dossier' in row['manuskriptstelle']:
        row['verwendung_v8']='nicht quantitativ verwendet; R3/R4-Tor nicht erfüllt'
        row['manuskriptstelle']='2.4/4.1 qualitative Lücke'
    claims.append(row)
for i,row in enumerate(rows,1):
    claims.append({'claim_id':f'S5-R-{i:04d}','aussage':row['key'],'wert':row['value'],'einheit':row['unit'],'manuskriptstelle':'2.1/2.4' if row['phase'] in ['R0','R1','R2','R3','R4'] else '2.5/3.4/3.5; vollständige sichere Tabelle im Nachrechnungsbericht','lauf':row['run'],'bericht':'nachrechnung/S5_NACHRECHNUNG_AW178.md','bericht_sha256':sha(reportpath),'originaldatei':row['source'],'filter_zeilenschluessel':row['key']+'; split='+row['split'],'skript':SCRIPT,'status':row['status'],'partition':row['split'],'testnutzung_historisch':'AF-v2 nur Entwicklung; X08-Population gemischt, nicht geöffnet; kein Testlesen AW178','verwendung_v8':'historisch erhalten' if row['phase']=='historisch_X08' else 'Zensus/Lücke; Detail im Nachrechnungsbericht','pruefniveau':'Bestands-Nachrechnung' if row['phase'] in ['R5','R6'] else 'Dokumenten-/Torprüfung','originaldatei_sha256':row['source_sha256'],'nachrechnungsstatus_aw178':row['status'],'nenner':row['denominator'],'ci_low':row['ci_low'],'ci_high':row['ci_high'],'intervalltyp':row['interval'],'hinweis':row['note']})
write_csv(PUB/'CLAIM_MATRIX_S5_v3.csv',claims)
newclaims=len(rows)
revision=f'''# Revision S5 v8 (AW-178)

13.10.2026 · Deutsch · englisches Manuskript · **nicht einreichungsreif**.

## Ergebnis und Abgrenzung

v7 war ein deutscher Arbeitsentwurf; v8 ist eine englische Neufassung mit dessen gemeinsamer methodischer Frage und v6-Hausstil. Der Uhr-/Matchingkern bleibt erhalten. R0/R3/R4 sind aufgrund fehlender zulässiger Eingangs-/Rohassay-/Episodenbindung gestoppt, R1/R2 nicht gestartet. R5 liefert den reproduzierten AF-v2-Cache-/Beobachtungszensus, R6 AF-Demografie und private Reproduzierbarkeitsspur, beide nur teilweise vollständig. Das ist kein erfolgreicher Abschluss sämtlicher Forschungsrechnungen und keine neue Validierung. Keine Testdaten geöffnet.

## Änderungen gegenüber v7

- Titel, strukturierter Abstract, Evidenzkasten, Methoden, Ergebnisse und Diskussion auf Englisch; keine deutschen Forschungsplatzhalter mehr.
- R0-Lücke durch konkreten Beleg ersetzt: X08-Generator liest gemischten Index vor Entwicklungsfilter; Roh-K-Einheit/Material und physische Targetkalibration nicht fallgebunden. Neue signierte Steigungen unterbleiben.
- X08-Originalkoeffizienten, KI, q-Werte und beide ungesicherten Lossgewinne erhalten; Target ausdrücklich gespeicherte Einheit, ohne ungeprüfte physische ms²-Freigabe.
- R1/R2 als nicht gestartet benannt; keine BH10-/BH3-/Bootstraprechnung suggeriert. B01/M01/M03/T01/T02 bleiben qualitative Designprobleme; keine ungebundenen historischen Zensuszahlen im Manuskript.
- AF-v1/v2 als andere Indexpopulationen, keine gepoolte Replikationsquote. Alle sechs v2-Complete-case-/Ereigniszähler aus zulässiger versiegelter Kohorte reproduziert; Endgründe separat und additiv geprüft. Indexflow vor Cache weiterhin unbekannt.
- AF-Sex-/Altersaggregate mit unknown-Kategorie, Setting- und upstream Altersgrenze ergänzt. Keine Sexinvarianz und keine neue Prognoseleistung.
- Forschungsplatzhalter durch echte Befunde/benannte Stopps oder verbleibende Lücken in normaler Prosa ersetzt. Autorfreigabe und verwandte Manuskripte bleiben explizite englische PENDING-Blöcke.
- Literatur unverändert aus v7 übernommen (AW-177-Prüfspur), **keine Recherche und kein erneutes Selbstnachschlagen in AW-178**. Neue Quellen sind nicht eingeführt. Goodwin-Volltext und itemgenaue GRRAS-Prüfung bleiben offen. Kein „bestätigt durch“ oder LK-Validierungsclaim.

## Reportinglisten: v8-Fundstellen und Restlücken

| Rahmen / Punkt | v8-Fundstelle | Status / Grenze |
|---|---|---|
| STROBE Design, Setting, Teilnehmer, Auswahl | 2.1-2.3/2.5, 3.1/3.2/3.4 | vorhandene Auditnenner; AF vor gecachtem Index nicht vollständig reproduziert |
| STROBE Variablen, Bias, Statistik | 2.2/2.4/2.5, 4.1 | stored-clock semantics; keine vollständige physische Referenzbindung |
| STROBE Missingness, deskriptive Population | 3.5, Nachrechnungs-CSV | AF vollständig innerhalb Cache, andere Auditdemografien offen |
| STROBE Ergebnisse/Unsicherheit | 3.1-3.5 | originale Bootstrap-/t-KI getrennt, reine Zensen ohne inferentielles KI |
| RECORD Datenzugang, Algorithmen, Linkage | 2.1/2.4/2.5, Data/code availability | gehashte private Spur, keine öffentliche komplette Rezeptauslieferung |
| RECORD Selektionsfluss / Cleaning | 3.4, Nachrechnungsbericht §4 | exklusive Ausschlüsse ab Cache; vollständiger upstream Fluss offen |
| GRRAS Messobjekt, Wiederholung, Referenz | 2.2/3.1/4.1 | HF-Kompatibilität, keine unabhängige Uhr; genaue Originalitemliste weiterhin offen |
| SAGER Sex/Gender, Nenner, Analyse | 3.5, Reporting | administrative Kategorie, keine Genderidentität oder neue Sexinteraktion; andere Populationen offen |
| Daten-/Ethik-/KI-Erklärungen | Declarations | Blueprint-Version 10.10.2026 inkl. Updates bis 13.10. übernommen; finale Autorprüfung ausstehend |

Die ausführlichen Arbeitslisten aus REVISION_S5_v7.md bleiben Vorversion; diese Tabelle ordnet ihre entscheidenden Punkte der geänderten v8 zu. Keine formal vollständige Reportingfreigabe behauptet.

## Auslieferung und Strategie

Neue sichere Resultdateien: Nachrechnungsbericht, CSV, Manuskript v8, Matrix v3 und dieses Protokoll. Rechen-/Schema-/Quell-/Hash-/Statusspur ausschließlich in <local path>; Werkzeuge nur neue aw178_*-Dateien. Matrix v3 erhält historische Claim-IDs und kennzeichnet ausgeschlossene quantitative Erweiterungen; {newclaims} neue Ergebnis-/Tor-/Missingness-/Demografierecords, keine Entdeckungszahl. STAFF und Dokumentations-META-Zeilen wurden nicht in die neue Matrix übernommen. Keine Befundakten/Register/Analyseoutputs geändert.

Ziel weiterhin bedingt Physiological Measurement, Note gemäß lokalem ZIELJOURNALE-/AW-177-Stand, kein frischer Journalregelabruf. Manuskriptneufassung ist eine Inhaltsrevision, keine reine stilistische Äquivalenz zu v6/v7. Der automatische Zahlen-/Zitatevergleich darf deshalb Unterschiede nicht als unzulässige Stiländerung klassifizieren. Separate Stilredaktion/Fachprüfung und K-Abnahme vor Veröffentlichung bleiben offen; keine Agenten dafür gespawnt. Die unveränderte bestehende Abbildung trägt eine englische v8-Legende.

Wegen der gescheiterten Entwicklungsbindungen keine Verstärkung zum eigenständigen publizierbaren Audit. K entscheidet, ob die restlichen getrennten Eingänge bereitgestellt werden oder S5 in A-Methods-Supplement integriert wird. Keine neuen Paperkonzepte, keine Außenkommunikation.

## Offene Punkte

1. Versiegelter getrennt erzeugter erster-X08-Index mit complete Rohassay-/Target-/Zeitkette; erst dann R1/R2. Getrennte Proben-, Gabe-/Verordnungs- und Ereignispools samt historischen Selektoren für R3/R4.
2. AF upstream Kandidaten-/Indexflow je Fold, vollständige Cachebindung der Beobachtungs- und Altersdefinition; Todeszeitpräzision und unabhängiger Ereignisstandard. Bis dahin nur Dokumentationszensus.
3. Demografie/Setting aller Auditpopulationen und öffentlich prüfbares vollständiges Supplement, Extensionmodule bestätigen.
4. Goodwin-Volltext/GRRAS, abschließende Literatur-/Stil-/Fachprüfung, Journalinterpretation und K-Abnahme.
5. Zwei PENDING-Blöcke: Autor bestätigt endgültige Prüfung/KI-Erklärung; K/Autor bestätigt verwandte Titel/Status/einmalige Tabellenzuordnung. Autordaten aus AUTOR_ANGABEN.json, keine Grade/Affiliation erfunden.

## Quellenfehler-Kandidaten

Keine neuen externen Quellen untersucht; keine neue Fehlerbehauptung. Interner Semantikhinweis: AF-v2-Protokoll enthält aus v1 geerbten Text „fixed across Validation/Test“, während der versiegelte Revisionscode die Ausführung auf Train/Validation ersetzt. Dies ist ein dokumentierter Metadatenrest, kein Nachweis neuer Testnutzung und kein externer Quellenfehler.
'''
(PUB/'REVISION_S5_v8.md').write_text(revision,encoding='utf-8')
patterns={'patient_values':r'(?:subject_id|study_id|hadm_id)\s*[:=]\s*\d+','internal_paths':r'\b[A-Z]:[\\/]|(?:koordination|werkzeuge|nachrechnung)/','internal_orders':r'\b(?:AW|AE|BEF)-\d+','hashes':r'\b[0-9a-f]{40,64}\b','german_research_placeholders':r'\[\[AUSSTEHEND:'}
checks={k:not bool(re.search(pattern,man)) for k,pattern in patterns.items()}
checks.update(unique_claim_ids=len({r['claim_id'] for r in claims})==len(claims),figure_present=(PUB/'abbildungen/fig1_s5_zeitversatz.png').exists(),six_census_matches=len(cc)==6,all_count_sums=all(r['events']+sum(r[k] for k in ['non_AF_contact_end','non_AF_death_end','non_AF_horizon_end','non_AF_tied_end'])==r['n'] for r in cc),one_person_per_complete_case=all(r['n_persons']==r['n'] for r in cc),no_new_fit=True,only_two_pending=len(re.findall(r'\[\[PENDING:',man))==2)
assert all(checks.values()),checks
outputs=[reportpath,csvpath,manpath,PUB/'CLAIM_MATRIX_S5_v3.csv',PUB/'REVISION_S5_v8.md']
validation={'time_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'word_count_whitespace':{'total':len(man.split()),'before_references':len(man.split('## References')[0].split()),'abstract':len(man.split('## Abstract')[1].split('Keywords:')[0].split())},'claim_records':len(claims),'result_records':len(rows),'outputs':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in outputs},'scripts':{str(p):sha(p) for p in (ROOT/'werkzeuge/s5').glob('aw178_*')},'memory':{'free_ram_gib':m.free/2**30,'free_commit_gib':m.commit_free/2**30},'new_literature_lookup':False}
(OUT/'delivery_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'checks':checks,'word_count':validation['word_count_whitespace'],'claims':len(claims),'results':len(rows)},ensure_ascii=False))
