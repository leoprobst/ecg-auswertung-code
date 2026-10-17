# Spezifikation S3 MC-MED v2.2: Kalium innerhalb der Person und Hyperkaliämie mit Bikarbonat in der Notaufnahme

Stand 16.10.2026, Entwurf durch ein KI-Werkzeug (Claude) im Auftrag des Autors. **Von der Code-Review (KI-Werkzeug, Runde 4) als bindungsreif bewertet; gebunden durch die Koordination des Autors.** Zeitpunkt, Sperrdatei und Hashes stehen in der Bindungsnotiz (`BINDING_NOTE.md`), nicht in dieser Datei (ihr Hash ist Teil der Bindung). v2.2 ersetzt v2.1 und behebt die drei Bindungshindernisse aus der Code-Review zu v2.1 (Runde 3, die Punkte 2 bis 5 der Runde 2 sind bestätigt); sonst gilt v2.1 wörtlich. v1, v2, v2.1 und die zugehörigen Skripte bleiben unverändert erhalten. v2.1 behob die fünf Muss-Punkte aus der Code-Review zu v2. Grundlage: die Machbarkeitsprüfung auf dem Trainingsteil, die Zweitmeinung zu v1 (KI-Werkzeug), die Code-Review zu v1 (KI-Werkzeug) und die Entscheidungen des Autors. Bis zur Bindung wurde **keine Assoziation Elektrolyt gegen ECG auf echten Daten gerechnet**; der Testsplit wurde weder geöffnet noch gehasht.

## 0b. Änderungen gegenüber v2.1 (Code-Review Runde 3, drei Bindungshindernisse)

| Nr. | Gegenstand | Änderung | Abschnitt |
|---|---|---|---|
| 1 | Verifizierte und verwendete Konfiguration | `chain(cfg)` liest `min_cover_s` und **alle Messparameter** (alle Werte von `aw197_messung.CFG`, `aw203_messung.V2`, der daraus abgeleitete Rauschfilter und die HF-Schwelle) aus der **übergebenen, gebundenen `cfg`** (`cfg['messkette']`), überträgt sie auf die Module und prüft die Übernahme; bei Wechsel der `cfg` wird neu übertragen. | 5 |
| 2 | Abhängigkeiten | **Feste Modulliste** `FESTE_MODULE` im Code (`s3_signal.py`, `aw197_messung.py`, `aw203_messung.py`); keine austauschbare Liste (Parameter `abhaengig` entfernt). Genau diese Pfade werden gehasht **und** importiert: Die Bytes werden gelesen, gehasht und genau diese Bytes ausgeführt; der Hash muss zur Laufzeit dem der Bindung entsprechen (sonst Abbruch vor der Ausführung), in Worker-Prozessen ebenso. Das von `aw203` verwendete `aw197` ist das gehashte Modul. | 5, 11 |
| 3 | Wiederaufnahme | Messungen und Erledigt-Marken tragen den **Bindungs-Hash** (Hash der Sperrdatei). Bei Wiederaufnahme werden nur Messungen mit **gleicher** Bindung verwendet; andere (auch ohne gespeicherte Bindung) werden in die Archivtabellen `mess_v2_archiv` und `mess_done_archiv` **verschoben** (mit Zeitstempel, nicht gelöscht) und neu gemessen. `laden` überschreibt den Freigabe-Hash eines Bestands nicht: `meta` ist ein fortlaufendes Protokoll; maßgeblich ist der letzte Eintrag. | 8, 11 |

## 0a. Änderungen gegenüber v2 (v2.1; Muss-Punkte der Code-Review, Runde 2)

| Nr. | Gegenstand | Änderung | Abschnitt |
|---|---|---|---|
| 1 | Freigabe | Produktiv **keine austauschbare Hashfunktion**: `pruefe_start`, `gesamtlauf` und `annotation_vorbereiten` haben keinen Parameter `hash_fn`; geprüft werden die Hashes der **tatsächlich verwendeten** Dateien und der **tatsächlich übergebenen `cfg`** (nicht der globalen `CFG`). Fehlt `SHA256SUMS.txt` oder schlägt eine Prüfung fehl, **bricht das Skript ab, bevor ein Besuch geladen wird**. | 3.2, 11 |
| 2 | Ergebniszeiten | Serum- und POC-Ergebniszeiten werden **nicht gemeinsam aggregiert**: Serum behält die Serum-Ergebniszeit (`r_s`), andere Quellen ihre eigene (`r_o`). Fehlt die Serum-Ergebniszeit, wird das Paar ausgeschlossen, **nicht** durch die POC-Zeit ersetzt. | 4.1 |
| 3 | Überlappung | Fensterüberlappung wird **überall** ausgeschlossen: auch in der Rohamplituden-Sensitivität (alle gültigen Paare einschließlich Polaritätswechsel). Für die Wiederholbarkeit wird die **tatsächliche Fensterüberlappung** der (an Segmentgrenzen verschobenen) Messfenster geprüft, nicht der Mittenabstand; die Wiederholbarkeits-Beschreibung schließt überlappende Fensterpaare zusätzlich aus und zählt sie. | 4.3, 5, 6.1 |
| 4 | Maske | **Explizite Liste der Zählfelder** (`ZAEHLFELDER`) statt Regex. Zeit- und Schwellenparameter (z. B. `rep_fenster_h`) werden nie maskiert. Ein Test prüft, dass jede ganzzahlige Ausgabegröße des Gesamtlaufs in der Liste steht. | 4.4, 11 |
| 5 | H4-AUC adjustiert | Die Regression auf Erstkalium und Wiederholungsabstand wird **innerhalb jeder Bootstrap-Ziehung neu geschätzt** (nicht Residuen einer einmaligen Schätzung resampelt). | 6.4 |

## 0. Ergebnis zuerst

1. **Eine einzige, vollständig vorab gebundene Gesamtanalyse über alle MC-MED-Splits** (Train, Val, Test; Zweitmeinung Punkt 3, Entscheidung Nr. 1). Kein Bestätigungsanspruch, kein Einfrieren, keine Entwicklung/Test-Trennung. Ergebnisse je Split sind eine ergänzende Stabilitätsbeschreibung. Nach der Bindung gibt es keine ergebnisabhängige Anpassung. Der Code öffnet den Testsplit nur im gebundenen Gesamtlauf; jede Datenfunktion verlangt eine Freigabe, die nur die Bindungsprüfung ausstellt.
2. **Kein scharfes Fallzahltor.** Schätzer, Intervalle und Kontrollen werden immer berichtet. Die Präzision wird mit Streuung der Kaliumänderung, Residualstreuung und Personenclustern geplant (Abschnitt 8). Die Residualstreuung lässt sich ohne Zielgrößen nicht schätzen; die Grenze ist offen benannt.
3. **Timing:** Abstände des ECG zu Auftrags- und Ergebniszeit werden berichtet; gleiche oder überlappende Messfenster an beiden Paarenden sind ausgeschlossen; D1 gegen D3 wird auf denselben Paaren verglichen; die Fensterfehler der Code-Review sind behoben (Abschnitt 4.3).
4. **Behandlung in drei Kategorien** (dokumentiert, unklar, kein Hinweis), Fenster über beide Laborzuordnungen und ECG-Messungen; Calcium ist ECG-wirksame Behandlung; Dialyse hat eine eigene Regel. Primär: kein Hinweis; Sensitivität: ohne dokumentierte Behandlung.
5. **Modell:** Paardifferenzen mit freiem Achsenabschnitt, unadjustiert und HF-adjustiert auf identischen Fällen, HF als Spline-Differenz f(HF₂) − f(HF₁), Huber-Sensitivität, Prüfung einflussreicher Paare, Schichtung nach Ausgangsbereich und Richtung, Messfehler-Szenarien, zensierte Kaliumwerte nie als exakte Werte.
6. **Endpunkt:** ausdrücklich die **QRS-relative T-Amplitude**; Polaritätsgruppen und Ausschlüsse sichtbar; die rohe vorzeichenbehaftete Ableitung-II-Amplitude als Sensitivität.
7. **H4 umbenannt:** persistierend erhöhtes gegen rasch normalisiertes Kalium ohne dokumentierte Behandlung; gleiches Wiederholungsfenster, erste Wiederholung, separate unklare Gruppe; kein Anspruch auf Pseudohyperkaliämie. 3 mm und AUC 0,65 sind beschreibende Marken ohne Entscheidungsanspruch.
8. **Kontrollen:** Querschnitts-PK und Wiederholbarkeit sind Beschreibungen, keine Tore; technische Ausschlüsse sind von biologischen Kontrollen getrennt; "Intervall enthält null" gilt nicht als Nullnähe; Kategorien sind überschneidungsfrei.
9. **Klinische Zielgröße:** mV und mm bei 10 mm/mV, dazu Wiederholungsgrenze und individuelle T-Änderung je Stufe der Kaliumänderung, gemessen an der Wiederholungsstreuung (Abschnitt 2).
10. **Code nach der Code-Review vollständig korrigiert** (Abschnitt 13), Selbsttest 40 von 40 (v2.2), einschließlich Grenzfällen aus der Code-Review und je einem gezielten Test pro Muss-Punkt beider Runden.
11. **Optional:** kaliumblinde Referenzannotation durch den Anwender an rund 100 zufälligen Blöcken (Abschnitt 10).

## 1. Rollen und Status

| Gegenstand | Rolle | Status |
|---|---|---|
| H1 Steigung QRS-relative T-Amplitude je Kalium | primär | eine gebundene Gesamtanalyse, **explorativ-bestätigend im Sinn einer vorab gebundenen Schätzung, nicht als Bestätigung einer Hypothese** |
| H2 T-Steilheit | sekundär | wie H1, eigenes Urteil |
| H3 QRS-Dauer gegen Bikarbonat bei K ≥ 6,5 | sekundär | die frühere Analyse in MIMIC (unpartitionierte Rohdaten) ist keine unabhängige Vorhersage |
| H4 persistierend erhöht gegen rasch normalisiert | sekundär, deskriptiv | kein diagnostischer Anspruch |
| PK Querschnitt, Wiederholbarkeit | Beschreibung | kein Tor |
| NK1, NK2, Natrium (Pipeline-Kontrolle) | biologische beziehungsweise rechnerische Kontrolle | Abweichung von der Nullnähe verbietet die inhaltliche Deutung von H1/H2 |

**Hypothesen nicht blind:** Die Erwartung für H1 stammt aus VitalDB (unveröffentlichte Analyse des Autors: 0,0213 mV je mmol/L, KI 0,0103 bis 0,0331), die für H3 aus einer früheren Analyse in MIMIC (−0,58 ms je mmol/L). MC-MED ist von beiden unabhängig. Evidenzstatus: vorab festgelegt, vor der Sicht auf Zielgrößen in MC-MED; Erwartungen nicht blind.

## 2. Klinische Frage und klinisch berichtbare Ergebnisgrößen

**Umrechnung:** 1 mV = 10 mm bei Standardverstärkung (25 mm/s); 0,010 mV = 0,1 mm; 0,30 mV = 3 mm (Funktion `mm`).

| Nr. | Klinische Frage | Technische Größe | Berichtete Ergebnisgrößen |
|---|---|---|---|
| **K1 (H1)** | Ändert sich die T-Welle am Notaufnahme-Monitor so stark mit dem Kalium, dass ein Arzt Änderungen erkennen kann? | β: Änderung der QRS-relativen T-Amplitude (mV) je mmol/L Kaliumänderung, Ableitung II, Paare ohne Behandlungshinweis, HF-adjustiert | (a) β in mV und mm je mmol/L mit Intervall; (b) erwartete T-Änderung in mm bei 1 und 2 mmol/L; (c) **Wiederholungsgrenze RC** = 1,96·√2·SD innerhalb (95 % der Differenzen zweier Messungen derselben Entnahme in zwei getrennten 60-s-Fenstern) in mV und mm, und das Verhältnis β zu RC; (d) **individuelle T-Änderung je Stufe der Kaliumänderung** (≤ −1,0; −1,0 bis −0,5; Betrag < 0,5; 0,5 bis 1,0; ≥ 1,0 mmol/L): Median und Quartile in mV und mm, Anteil mit Betrag über RC, Anteil gleichgerichtet über RC (Stufen unter 10 Paaren ohne Anteile); (e) Residualstreuung der T-Änderung bei gegebener ΔK |
| K2 (H2) | Wird der absteigende T-Schenkel mit dem Kalium steiler? | β der T-Steilheit (mV/s je mmol/L) | mV/s je mmol/L; keine mm-Umrechnung |
| K3 (H3) | Ist der QRS-Komplex bei sehr hohem Kalium breiter, wenn das Bikarbonat niedrig ist? | β der QRS-Dauer (ms) je mmol/L Bikarbonat, K ≥ 6,5, Serum, adjustiert auf Kalium und Alter | ms je mmol/L; QRS-Mittel je Bikarbonat-Terzil; nur Gruppenmittel (Messrauschen der QRS-Dauer rund 20 ms). **Niedriges Bikarbonat ist ein Surrogat und kein Beweis einer Azidose** (Kompensation, Mischstörungen) |
| **K4 (H4)** | Hilft das Monitor-ECG bei einem hohen Kaliumwert, einen falsch hohen Wert zu erkennen? | Unterscheidung persistierend erhöhten (PE) von rasch normalisiertem (RN) Kalium ohne dokumentierte Behandlung anhand der T-Amplitude am Erstwert | AUC mit Personen-Clusterintervall, AUC adjustiert für Erstkalium und Wiederholungsabstand; Anteil PE mit T ≥ 0,30 mV (3 mm) und Anteil RN mit T < 0,30 mV, je mit Intervall. **Beschreibende Marken ohne Entscheidungsanspruch**; kein Pseudohyperkaliämie-Anspruch |

**Aussagegrenzen (verbindlich in jeder Berichtssprache):**
* Die Wiederholungsgrenze sagt, wie groß eine Änderung sein muss, um über dem Messrauschen derselben Entnahme zu liegen. Sie ist **keine** individuelle Nachweisgrenze für eine Kaliumänderung und **keine** Aussage über die visuelle Erkennbarkeit am Monitor. Weder 1 mm noch Stufenanteile belegen Erkennbarkeit; der Quotient 0,1 mV/β (v1) entfällt.
* Die Wiederholungsstreuung umfasst Signalrauschen und kurzfristige physiologische Schwankung zwischen zwei Fenstern derselben Entnahme (Minuten), nicht die Variation über Stunden.
* H1 bis H3 sind Aussagen über Gruppen von Paaren.

**Erwartung vorab:** Bei 0,21 mm je mmol/L (VitalDB) ist die Antwort auf K1 am Einzelpatienten voraussichtlich nein; der Gruppenzusammenhang kann dennoch nachweisbar sein. Die Spezifikation legt beide Größen fest, bevor Daten gesehen werden. Die Marke 0,30 mV (3 mm) für K4 ist eine beschreibende Setzung aus dem Messbereich der Machbarkeit (mediane T-Amplitude 0,19 bis 0,22 mV) ohne Datenoptimierung (kein Youden-Index) und ohne belegte Entscheidungsrelevanz.

## 3. Machbarkeitssicht, Eingaben, Zeit- und Einheitenvertrag

### 3.1 Was vor der Bindung gesehen wurde (offengelegt)

Die Machbarkeitsprüfung lief **nur auf Train** (`split_random_train`) und zählte: Laborverfügbarkeit, Paare (konsekutive Serumpaare), zeitliche ECG-Abdeckung (D1/D2/D3), Signalgüte der 10-s-Blöcke, Gültigkeit und Wiederholbarkeit der Messkette aw203 v2 in 60- und 300-s-Fenstern (die Zahlen dieser Zählungen sind hier nicht wiedergegeben), **marginale Verteilungen von Zielgrößen** (T-Amplitude Median 0,19 bis 0,22 mV; Streuung der T-Änderung über alle Paare SD 0,127 mV; Streuung der QRS-Änderung), Behandlungsanteile, HF-Änderung und die Zahl der Extremwerte. **Nicht gesehen:** jede Beziehung zwischen Kalium (oder einer anderen Exposition) und einer ECG-Größe, Querschnitt oder innerhalb der Person, die Gruppen BH/VP beziehungsweise PE/RN mit T-Amplituden, jede Nullkontrolle. Bekannt sind außerdem die externen Ergebnisse (VitalDB und die frühere Analyse in MIMIC). Val und Test wurden für die Machbarkeit nicht gelesen.

### 3.2 Eingaben

MC-MED 1.0.1 (PhysioNet, Credentialed Health Data License 1.5.0), lokal, nie an Cloud-Dienste oder KI-Werkzeuge. Prüfsummen gegen `SHA256SUMS.txt` (am 16.10.2026 für alle außer Test geprüft; im Gesamtlauf prüft das Skript vor dem Laden irgendeines Besuchs alle Eingaben **einschließlich** des Testsplits, `eingaben_pruefen`; **fehlt `SHA256SUMS.txt` oder weicht ein Hash ab, bricht das Skript ab**):

| Datei | SHA-256 |
|---|---|
| `data/labs.csv` | 254ab541f9dc61c175c177dc8d53b9839a37b7f8cb6ef64a200574ba32c31611 |
| `data/orders.csv` | d4521018baac4bb85d4a15098bb176a2aeaace81e042abbad0d27a1f036822d9 |
| `data/visits.csv` | 9fea50d6ffc35affb827846cdfc4fc2f3bd23463a1d38192cb3de3c1e9552e86 |
| `data/numerics.csv` | 225509cec1822e7b89bb047837775c90cf40d1aa8136bd94eb85330782d66ab9 |
| `data/split_random_train.csv` | b662959b0ac6f3ef99ac3cfa23f84d66b4b16d57c24da4d2ebb2ab091463c88e |
| `data/split_random_val.csv` | 7430416314733695dd3496201b2c6fabac05f07f8c95ff5edb163f238d2dd397 |
| `data/split_random_test.csv` | im Gesamtlauf gegen `SHA256SUMS.txt` geprüft (vor der Bindung nicht gehasht) |
| Wellenformen Ableitung II | vor der Bindung wie S5 (`wellenhash`) zu prüfen; Pleth und Resp werden nicht verwendet |

Splits: `split_random_*`, alle drei im Gesamtlauf.

### 3.3 Zeit- und Einheitenvertrag (Konstante `VERTRAG`, Teil der Bindung, im Lauf geprüft)

| Größe | Vertrag | Prüfung |
|---|---|---|
| Zeit | UTC-Sekunden seit 1970; CSV-Zeitstempel (Suffix Z) und WFDB-Kopfzeit (Uhrzeit, Datum) beide als UTC gelesen | `zeitvertrag_pruefen`: Median(erster Monitor-HF-Zeitstempel minus erster Segmentbeginn) muss zwischen −60 und 300 s liegen (S5-Referenz rund 40 s), mindestens 20 Besuche; sonst Abbruch |
| Signal | Ableitung II in mV | `einheit_pruefen` je Datei (Kopfzeile units muss mV sein); abweichende Segmente werden ausgeschlossen und gezählt |
| Kalium, Bikarbonat, Natrium | mmol/L | nur Zeilen mit Einheit mmol/L; abweichende Zeilen ausgeschlossen und gezählt (`kalium_zeilen_andere_einheit`) |
| HF | 1/min (Monitor) | Wertebereich nicht verändert |

## 4. Population, Entnahmen, Paare, Zeitbezug

### 4.1 Entnahmen und Paare

* **Entnahme:** ein Kaliumwert je Besuch, Quelle und Auftragszeit (`Order_time`). Serum/Plasma (`POTASSIUM`) bleibt auch bei gleichzeitigem i-STAT-Wert Serum (die Quelle wird nicht über `bool_and` überschrieben); "alle Quellen" bevorzugt Serum und nimmt sonst den eindeutigen i-STAT- oder Vollblutwert. Widersprüchliche Werte derselben Quelle und Auftragszeit entfallen.
* **Zensierte Werte (`>`, `<`)** tragen die Marke `cens`, werden nie als exakte Werte analysiert und schließen das Paar aus allen Modellen aus (Zahl wird berichtet).
* **Paar:** Entnahme und die zeitlich nächste folgende Entnahme desselben Besuchs, Abstand der Auftragszeiten 1 bis 12 h (konsekutiv). Bei Abstand unter 1 h oder über 12 h entsteht kein Paar.
* **Ergebniszeit quellengetreu:** Jede Entnahme trägt die Ergebniszeit **ihrer Quelle**: Serum die frühere Serum-Ergebniszeit (`r_s`), i-STAT/Vollblut die eigene (`r_o`); Serum- und POC-Zeiten werden nicht gemeinsam aggregiert (ein POC-Ergebnis nach 5 min darf nicht die Serumzeit von 90 min ersetzen und D3 sowie das Behandlungsfenster verändern).
* **Ohne Ergebniszeit** an einem Ende: Das Paar wird **ausgeschlossen**, die Ergebniszeit wird weder ergänzt (v1: 60 Minuten) noch durch die Zeit einer anderen Quelle ersetzt.
* **Zeitsemantik:** MC-MED enthält keine Abnahmezeit, nur Auftrags- und Ergebniszeit. Jede positive Aussage trägt den Satz: "Die Abnahmezeit ist nicht dokumentiert (nur Auftrag und Ergebnis); der Befund gilt für das ECG nächst der Auftragszeit innerhalb des Fensters Auftrag minus 15 min bis Ergebnis plus 15 min."

### 4.2 Behandlung: drei Kategorien

Quelle: `orders.csv` (Medikation mit `First_admin_time`; Dialyse-Aufträge der Pflege ohne Gabezeit nur mit `Order_time`). Klassen: *kaliumsenkend* = Altinsulin, Glukosebolus (D50, D20), Bikarbonat i.v., Albuterol, Kaliumbinder; *ECG-wirksam* = **Calcium i.v.** (wirkt auf das ECG ohne Kaliumsenkung, zählt als Behandlung); *Kaliumgabe* = KCl, Kaliumphosphat; *Dialyse* = Hämodialyse, Peritonealdialyse, CRRT, Dialysat (nicht "Dialysezugang", nicht Nephrologie-Konsil); *schwach/unklar* = andere Insuline, Schleifendiuretika. Magnesium i.v. wird nicht verwendet.

Fenster je Paar: **S = min(Auftrag a, ECG-Mitte a)**, **E = max(Auftrag b, Ergebnis b, ECG-Mitte b)**, alle Ränder einschließlich (≥, ≤). Vorlaufsetzungen: 120 min (dokumentiert) und 360 min (unklar), Setzung S4.

| Kategorie | Regel |
|---|---|
| **dokumentiert** | Gabe (First_admin_time) einer Klasse aus kaliumsenkend, ECG-wirksam, Kaliumgabe, Dialyse in [S − 120 min, E]; oder Dialyse-Auftrag in [S − 360 min, E] (Auftragszeit als Näherung für den Dialysebeginn) |
| **unklar** (und nicht dokumentiert) | Gabe einer dieser Klassen in [S − 360 min, S − 120 min) (fortwirkend); oder Auftrag ohne Gabezeit in [S − 120 min, E]; oder Gabe anderer Insuline/Schleifendiuretika in [S − 120 min, E]; oder Dialyse-Auftrag irgendwo im Besuch außerhalb des Fensters |
| **kein Hinweis** | sonst |

Die fehlenden Gabezeiten (34 % der Medikationsaufträge, bei D50 79 %) machen "kein Hinweis" zu "keine dokumentierte oder anzunehmende Behandlung", nicht zu "unbehandelt". **Primär:** kein Hinweis und Erstwert < 6,0. **Sensitivitäten:** kein Hinweis oder unklar (ohne dokumentierte Behandlung); alle Erstwerte. **Dokumentiert** bildet das Behandlungsstratum (nur beschreibend; Modell ab 100 Paaren).

### 4.3 Zeitfenster und Blockwahl

* **D3 (primär):** [früherer Zeitpunkt aus Auftrag und Ergebnis − 15 min, min(späterer Zeitpunkt + 15 min, früherer Zeitpunkt + 3 h)] (Obergrenze genau 3 h nach dem früheren, v1 hatte 3 h + 15 min; bei Ergebnis vor Auftrag, wie bei i-STAT, gilt der frühere der beiden). **D1 (Sensitivität):** [Auftrag − 15 min, Auftrag + 15 min], mit **eigener globaler Suche** über alle Segmente (nicht im D3-Zwischenspeicher).
* **Begründung D3 (Koordination):** Die Abnahmezeit ist unbekannt und liegt zwischen Auftrag und Ergebnis (nachgetragene Aufträge bei Point-of-Care zeigen, dass auch das nicht garantiert ist; Zweitmeinung: D3 ist eine Zuordnungsregel, kein validiertes Entnahmefenster). Gewählt wird der **saubere 10-s-Block (Raster 5 s) mit dem kleinsten Abstand der Blockmitte zur Auftragszeit**, global über alle Segmente. Der Preis ist Zeitunschärfe; deshalb werden die Abstände zu Auftrags- und Ergebniszeit berichtet (`timing`: Quantile, Anteil vor Auftrag, Anteil nach Ergebnis, je Paarende) und D1 gegen D3 auf denselben Paaren verglichen (`d1_gleiche_paare`).
* **Sauberer Block** (`block_gut` aus `s3_signal.py`, unverändert): kein fehlender Wert, keine Sättigung (> 4,5 mV), keine Flachlinie (≥ 60 ms), Spannweite ≥ 0,3 mV, 5 bis 37 Schläge, mittlerer RR 0,27 bis 2,0 s, Medianschlag ≥ 0,15 mV, Schlag-Schablonen-Korrelation (Median ≥ 0,90, ≥ 80 % der Schläge ≥ 0,80).
* **Messfenster:** 60 s um die Blockmitte, an die **tatsächlichen Segmentgrenzen** verschoben und frisch aus dem Segment gelesen (v1 verschob an den Grenzen des D3-Ausschnitts); Segmente kürzer als 60 s liefern keine Messung.
* **Keine Kollision, überall:** Überlappen die Messfenster der beiden Paarenden (oder sind sie gleich), wird das Paar ausgeschlossen (`kollision`); Zahl berichtet. Das gilt für die Primärmenge, alle Sensitivitäten **einschließlich der Rohamplituden-Sensitivität mit Polaritätswechsel** und NK1.
* **Zweites Fenster (Wiederholbarkeit):** nächster sauberer Block mit Mittenabstand ≥ 90 s zur ersten Mitte (absolute Zeit, **über alle Segmente**), **dessen tatsächliches, an die Segmentgrenzen verschobenes Messfenster das erste Messfenster nicht überlappt** (`fenster_ausser`). Der Mittenabstand allein genügt nicht: Bei Mitten 5 s und 95 s in einem 105-s-Segment ergeben sich die Fenster [0, 60] und [45, 105].

### 4.4 Zählkette und technische Ausschlüsse

Das Skript berichtet sequenziell: Paare konsekutiv → ohne Ergebniszeit ausgeschlossen → zensiert ausgeschlossen → ECG zeitlich an beiden Enden → Messung gültig an beiden Enden → Polarität sicher und gleich → keine Fensterüberlappung → HF an beiden Enden → davon Behandlung dokumentiert/unklar → Behandlung zugelassen → davon Erstwert ≥ 6,0 → **Primärpaare** (Zahl und Personen). Zusätzlich die Polaritätsgruppen (++, −−, gemischt, unsicher). Die Kleinzellenregel wird über die **explizite Liste `ZAEHLFELDER`** angewendet (Werte 1 bis 9 unter diesen Schlüsseln, auch als Gleitkomma, werden "<10"); Zeit- und Schwellenparameter bleiben unmaskiert. Komplemente werden vergröbert: Ist der Abgang zwischen zwei Schritten 1 bis 9, wird der spätere Wert (und folgende gleiche Werte) auf Zehner abgerundet und markiert (`zaehlkette_vergroebern`). **Technische Ausschlüsse (diese Kette) sind von den biologischen Kontrollen (Abschnitt 7) getrennt.**

**ΔHF:** Monitor-HF (Median ±5 min um die Blockmitte, mindestens 3 Werte) je Paarende.

## 5. Messkette aw255-v3 und Endpunkt

**Bindung:** feste Modulliste `n3/aw197_messung.py` und `aw203_messung.py` (aw203 v2, unverändert), `s3_signal.py`, dazu `s3_auswertung_v2_2.py`; HF-Rauschschwelle 0,0789531424618521 mV (99. Perzentil der Schläge von 100 zufälligen Train-Besuchen, ohne Kaliumbezug). **Alle Messparameter stehen in `cfg['messkette']`** (und damit im Konfigurationshash): `aw197` (rr_min 0,3; rr_max 2,0; ecto 0,20; med_n 11; **min_cover_s 20**; irr_frac 0,30; beat_pre 0,45; beat_post 0,55; hf_hp 50; hf_pre 0,25; hf_post 0,45; sat_abs 4,5; flat_ms 40; pace_slew 300; pace_frac 0,30; min_qrs_amp 0,15; pad_s 3,0) und `aw203_v2` (pre_s 0,200; post_rr 0,75; ts_k 0,18; ts_rr 0,25; te_k 0,50; te_rr 0,60; slope_k 0,12; thr_onset 0,10; min_r 0,10; min_tile 2500), gleich den Standardwerten der Module bis auf die Abweichung von aw203: Mindestabdeckung gültiger Schläge 20 s statt 60 s (aw203 maß in 5-min-Fenstern). `chain(cfg)` überträgt diese Werte auf die Module und prüft sie.

**Polarität:** Medianschlag (Bandpass 0,5 bis 40 Hz, Schlagorte aus der QRS-Hüllkurve); Hauptrichtung = größere Auslenkung gegen die Grundlinie; **unsicher**, wenn die kleinere mindestens 70 % der größeren beträgt. Das Signal wird so gedreht, dass die Hauptrichtung positiv ist, dann misst aw203 v2.

**Endpunkt (ausdrücklich): QRS-relative T-Amplitude** `t_amp` (mV; positiv = T gleichgerichtet zum QRS; Basis: PR-Isoelektrik des R-ausgerichteten Medianschlags). Sie ist **nicht** die vorzeichenbehaftete Ableitung-II-Amplitude. Gleichbleibende Polarität beweist keine konstante Elektrodenlage, wechselnde Polarität keinen Elektrodenwechsel; Morphologieänderungen können biologisch sein. Deshalb: Polaritätsgruppen und Ausschlüsse werden berichtet, und die **rohe vorzeichenbehaftete Ableitung-II-Amplitude** `t_roh = Polarität × t_amp` läuft als Sensitivität (a) auf den Primärpaaren, (b) auf allen gültigen Paaren einschließlich Polaritätswechsel und unsicherer Polarität. T-Steilheit `steil` = −(Steilheit des absteigenden Schenkels).

**Beschreibungen (keine Tore):**
* **Querschnitt (PK):** Steigung `t_amp` je Kalium zwischen Personen, eine feste Zufallsentnahme je Person (höchstens 6.000), adjustiert auf Alter, Geschlecht, HF. Zwischen- und Inner-Personen-Effekte können verschieden sein; kein Beleg für den Personeneffekt.
* **Wiederholbarkeit (WH):** zwei Messfenster derselben Entnahme **ohne tatsächliche Fensterüberlappung** (überlappende Paare werden ausgeschlossen und gezählt), gleiche sichere Polarität; SD innerhalb, ICC(1,1), Bias, RC. Wiederholbarkeit beweist keine Messrichtigkeit; die kaliumblinde Referenzannotation (Abschnitt 10) ist die ergänzende Prüfung.

## 6. Hypothesen, Modelle, Kategorien

### 6.1 H1 (primär)

**Modell:** Paardifferenzen, ΔT = α + β·ΔK + Σ γ_j·dB_j + ε, **freier Achsenabschnitt α** (mittlere zeitliche Drift), gewöhnliche Kleinste-Quadrate, Intervalle durch **Personen-Clusterbootstrap** (2.000 Ziehungen, Seed 255001, Perzentilintervall 95 %). dB_j = j-te Komponente von f(HF₂) − f(HF₁) mit f = **eingeschränkter kubischer Spline** (Harrell) mit **festen Knoten 55, 70, 85, 110/min** (drei Spalten; HF auf 30 bis 220/min begrenzt; Setzung S5). **Unadjustiert (nur α, β) und adjustiert laufen auf identischen Fällen.** HF-Adjustierung definiert eine HF-bedingte Assoziation; HF kann Vermittler einer Kaliumwirkung sein, deshalb sind beide Schätzer zu berichten.

**Begründung der Wahl (gegenüber einem gemischten Modell als Primärmodell):** Die Differenz entfernt zeitkonstante Unterschiede der T-Wellen-Form zwischen Personen (die Streuung zwischen Entnahmen ist größer als die innerhalb einer Person). Es gibt rund 1,07 Paare je Person; ein zufälliger Achsenabschnitt je Person ist kaum identifizierbar. Der Perzentil-Bootstrap hängt nicht an einer Normalannahme der schwerschwänzigen Differenzen. Gleiches Vorgehen wie in der VitalDB-Analyse. Das **gemischte Modell** (zufälliger Achsenabschnitt, REML, standardisierte Kovariaten, Optimierer lbfgs/bfgs/powell) läuft als Sensitivität; **Konvergenz und Vorzeichenübereinstimmung mit dem Primärschätzer werden geprüft, Warnungen und Fehler protokolliert** (nicht unterdrückt); scheitert es, entfällt die Sensitivität sichtbar.

**Kategorien (überschneidungsfrei; β in mV je mmol/L, Grenze G = 0,010 = 0,1 mm je mmol/L; Intervall [lo, hi]):**

| Kategorie | Bedingung |
|---|---|
| `negativ` | hi < 0 (entgegen der Erwartung; **nach technischer Prüfung interpretierbar**, nicht als Fehler verworfen) |
| `positiv_ueber_grenze` | 0 < lo und lo ≥ G (Effekte unterhalb der Grenze ausgeschlossen) |
| `positiv_unter_grenze` | 0 < lo und hi < G (nachweisbar, aber Effekte ab der Grenze ausgeschlossen) |
| `positiv_grenze_offen` | 0 < lo < G ≤ hi |
| `grenze_ausgeschlossen` | lo ≤ 0 ≤ hi < G (kein Nachweis; Effekte **ab der Grenze** ausgeschlossen) |
| `unpraezise` | lo ≤ 0 und hi ≥ G |
| `nicht_beurteilbar` | weniger als 30 Paare oder 10 Personen (einzige Schätzbarkeitsgrenze, kein Tor) |

**Sensitivitäten (im Gesamtlauf):** Huber-M-Schätzer (robust gegen Ausreißer); einflussreiche Paare (|DFBETAS| der Steigung > 2/√n; Schätzer ohne sie); Schichtung nach Ausgangskalium (≤ 3,5; 3,5 bis 5,0; > 5,0; Regression zur Mitte, Messfehler im Ausgangskalium gehen in ΔK ein); nur Anstiege, nur Abfälle; K₁ als Kovariate; Δ Bikarbonat; nur |ΔHF| < 10; eine Beobachtung je Person; gemischtes Modell; ohne dokumentierte Behandlung; alle Erstwerte; rohe T-Amplitude (a/b, Abschnitt 5); D1 gegen D3 auf denselben Paaren; alle Kaliumquellen; **Kaliummessfehler-Szenarien** (SD 0,1 und 0,2 mmol/L je Messung: Verdünnungsfaktor λ = 1 − 2·SD²/Var(ΔK), korrigierte Steigung β/λ, rein deterministische Skalierung); Stabilität je Split (Abschnitt 8). Sensitivitäten auf einer Teilmenge nutzen 500 Bootstrap-Ziehungen.

### 6.2 H2

Steilheit S positiv gerichtet, Modell und Kategorien wie H1, Grenze G = 0,2 mV/s je mmol/L (0,010 mV mal das VitalDB-Verhältnis Steilheit/Amplitude 20,8 je s; Setzung S6).

### 6.3 H3

Entnahmen mit K ≥ 6,5, **nur Serum, nicht zensiert**, gültige QRS-Dauer, Bikarbonat aus derselben Probe, eine (die erste) Entnahme je Person, mindestens 30 Personen. Modell: QRS-Dauer (ms) ~ Bikarbonat + K + Alter; Personen-Clusterbootstrap; Erwartung negativ (frühere Analyse in MIMIC: −0,58 ms je mmol/L); Kategorien gespiegelt mit G = 0,29 ms je mmol/L (die Hälfte davon). QRS-Mittel je Bikarbonat-Terzil. pH wird nicht verwendet (weniger als 10 Entnahmen mit pH innerhalb ±1 h).

### 6.4 H4 (deskriptiv)

Population: Erstwert ≥ 6,0 (Serum, nicht zensiert), gültiges ECG mit sicherer Polarität am Erstwert, Ergebniszeiten vorhanden. **Gleiches Wiederholungsfenster für alle:** die **erste** spätere Entnahme desselben Besuchs mit Abstand **0,25 bis 6 h** (vorab festgelegt; Setzung S7). Begründung der unteren Grenze: Wiederholungen unter 15 min sind wahrscheinlich Auftragsdopplungen derselben Probe; eine Mindestgrenze von 1 h (wie bei H1) würde gerade die rasche Nachkontrolle nach auffälligem Wert ausschließen.

| Gruppe | Definition |
|---|---|
| **PE** (persistierend erhöht) | Behandlungskategorie kein Hinweis und Wiederholung ≥ 5,5 |
| **RN** (rasch normalisiert) | kein Hinweis, Wiederholung < 5,0, Abfall ≥ 1,0 |
| **unklar** (separat berichtet) | Wiederholung 5,0 bis 5,5, oder Behandlungskategorie unklar |
| **dokumentiert** | dokumentierte Behandlung (separat, nicht verglichen) |

Kennzahlen: AUC (PE größer als RN) mit Personen-Clusterintervall; AUC der Residuen nach Regression der T-Amplitude auf Erstkalium und Wiederholungsabstand, wobei die **Regression in jeder Bootstrap-Ziehung neu geschätzt** wird (Schätzunsicherheit der Adjustierung geht in das Intervall ein); Anteil PE mit T ≥ 0,30 mV und Anteil RN mit T < 0,30 mV; Sensitivität der Zuordnung (unklare nach Folgewert zugeordnet); mediane T-Amplitude je Gruppe. Mindestens 30 je Gruppe, sonst nicht beurteilbar. Kategorie: `beschrieben`. AUC 0,65 und 3 mm sind beschreibende Marken (`auc_marke_erreicht`, `auc_intervall_ueber_0.5`), keine Entscheidungsgrenzen.

**Ehrliche Grenze:** Die Gruppen sind **Verlaufsgruppen, keine diagnostisch verifizierten Zustände**. Sie vermischen echte Änderung, Regression zur Mitte, Probenfehler und unerfasste Behandlung (Dialyse, undokumentierte Gaben, Spontanverlauf); ein echtes, rasch behandeltes Hyperkalium kann als RN erscheinen. Es gibt keinen Hämolyseindex und keinen Referenzstandard; auch i-STAT/Vollblut gilt nicht pauschal als hämolysefrei. Pseudohyperkaliämie bleibt eine mögliche Erklärung. Die Abnahmezeit ist unbekannt.

## 7. Kontrollen

**Technische Ausschlüsse** (Abschnitt 4.4) gelten für alle Analysen; **biologische/rechnerische Kontrollen** sind getrennt:

| Kontrolle | Aufbau | Bewertung |
|---|---|---|
| **NK1** zeitversetzte ECG-Fenster | Labor unverändert; beide ECG-Messungen durch einen zufälligen sauberen Block ersetzt, der 2 bis 6 h von der Auftragszeit entfernt liegt, dessen **tatsächlich gewähltes** Messfenster außerhalb von D3 liegt (20 Versuche, feste Ziehung); gleiche Paare wie die Primärmenge, soweit beide Messungen gültig, sicher, gleichpolig und ohne Fensterüberlappung sind; ΔHF aus denselben Blöcken | **keine Mindestzahl 100**; ab 30 Paaren Schätzung mit Halbbreite des Intervalls (Präzisionsangabe), 10 bis 29 nur Zahl; Nullnähe nach Abschnitt unten. Grenze: persistierende Physiologie kann echte Zusammenhänge behalten |
| **NK2** Paare verschiedener Personen | (ΔT, HF-Spline-Differenz) einer Paarzeile einer anderen Person zugeordnet; **feste Vollstichprobe** (Personenblöcke in zufälliger Reihenfolge, zyklische Verschiebung um mindestens die größte Blockgröße, 500 Permutationen, Seed 255003, nie die eigene Person) | prüft die Rechenpipeline (Kalibrierung), keine Physiologie, kein Falsifikationstest für H1 |
| **Natrium (Pipeline-Kontrolle)** | ΔT je 5 mmol/L Δ Natrium (Panelwert derselben Auftragszeit), HF-adjustiert; bei \|r(ΔNa, ΔK)\| > 0,30 partiell auf ΔK | **keine gesicherte Nullexposition** (Kopplung an Kalium über Behandlung, Dialyse, Infusionen); nur Pipeline-Kontrolle |

**Nullnähe nur über Äquivalenzgrenzen:** `nahe_null` = Intervall innerhalb ±0,005 mV je mmol/L (NK1, NK2) beziehungsweise ±0,005 mV je 5 mmol/L (Natrium); `von_null_verschieden` = Intervall schließt null aus; sonst `unpraezise`. **"Intervall enthält null" gilt allein nicht als Nullnähe.** Eine Kontrolle `von_null_verschieden` macht H1/H2 **nicht deutbar** (Schätzer und Intervalle werden berichtet, nicht gedeutet); `unpraezise` verbietet keine Deutung, wird aber neben ihr berichtet.

## 8. Gesamtlauf, Stabilität, Präzision

* **Gesamtlauf** (`gesamtlauf`): Bindung prüfen → Eingaben gegen `SHA256SUMS.txt` → alle Splits laden → Zeitvertrag prüfen → Entnahmen messen (wiederaufnehmbar **nur mit gleicher Bindung**: vorhandene Messungen mit anderer oder fehlender Bindung werden in `mess_v2_archiv`/`mess_done_archiv` verschoben und neu gemessen; die Zahl archivierter Messungen steht im Lauf) → Tabellen → `gesamt` → maskierte Ausgabe. Kein Modus ohne Bindung; die Bindungsprüfung stellt die einzige Freigabe aus.
* **Stabilität je Split:** dieselbe HF-adjustierte Steigung für Train, Val, Test getrennt (500 Ziehungen), **kein Bestätigungsanspruch**. Die Splits sind zufällig nach Besuch und nicht unabhängig bezüglich Personen.
* **Kein Fallzahltor.** Unter 30 Paaren oder 10 Personen: `nicht_beurteilbar`; sonst immer Schätzer, Intervall, Kategorie.
* **Präzisionsplanung (neu begründet).** SE(β) ≈ σ_res / (SD_ΔK · √(n / deff)); Halbbreite 1,96·SE; kleinste bei 80 % Power nachweisbare Steigung 2,8·SE (Funktion `praezisionsplanung`). Eingaben: **SD_ΔK ≈ 1,34 mmol/L** (Machbarkeit, saubere D1-Paare; Exposition, keine Zielgröße), **deff ≈ 1,07** (Paare je Person), **n** = Primärpaare in allen Splits (Schätzung unten) und **σ_res**. Die Residualstreuung nach Abzug des Kaliumeffekts lässt sich **nicht aus der Machbarkeit schätzen, ohne Zielgrößen im Zusammenhang mit dem Kalium zu sehen**; bekannt sind nur die marginale SD der T-Änderung (0,127 mV; enthält das Signal, also obere Schranke) und das Rauschen aus VitalDB (als untere Schranke der Differenz-SD, ≥ 0,06 mV). **Grenze offen: σ_res liegt zwischen etwa 0,06 und 0,127 mV.** n: aus den sauberen D3-Paaren in Train (Zählung vor der Bindung), davon Paare mit Behandlungsstatus ohne Hinweis (alte Definition ohne "unklar") und Erstwert < 6,0, mit der erwarteten Gültigkeit unter v3, abzüglich unklarer Behandlung und Fensterüberlappung, hochgerechnet auf alle Splits: **etwa 200 bis 330 Primärpaare**.

| σ_res (mV) | n = 150 | 200 | 260 | 330 |
|---|---|---|---|---|
| 0,06 | Halbbreite 0,007; MDB 0,011 | 0,006; 0,009 | 0,006; 0,008 | 0,005; 0,007 |
| 0,09 | 0,011; 0,016 | 0,010; 0,014 | 0,008; 0,012 | 0,008; 0,011 |
| 0,127 | 0,016; 0,022 | 0,014; 0,019 | 0,012; 0,017 | 0,011; 0,015 |

(mV je mmol/L; Halbbreite des 95-%-Intervalls; MDB = kleinste bei 80 % Power nachweisbare Steigung.) Gegen die VitalDB-Steigung 0,0213 und die Grenze G = 0,010 ergibt das: Bei σ_res ≥ 0,09 ist `positiv_ueber_grenze` nur bei einer Steigung deutlich über G erreichbar, bei kleiner Streuung und n ≥ 260 realistisch; `positiv_grenze_offen` oder `unpraezise` sind plausible Ergebnisse. Die beobachtete Präzision (SE, Halbbreite, Residual-SD, Paare je Person) wird im Lauf berichtet.

## 9. Deutungstabelle

| Ergebnis | zulässige Aussage |
|---|---|
| Primärmenge nicht schätzbar (< 30 Paare oder < 10 Personen) | `nicht_beurteilbar`; Beschreibung der Zählkette |
| Kontrolle `von_null_verschieden` (NK1, NK2 oder Natrium) | H1/H2 nicht deutbar: Artefakt (Messkette, Selektion, Behandlung) vorrangig; Schätzer werden berichtet |
| H1 `positiv_ueber_grenze` | Innerhalb der Person geht eine Kaliumänderung mit gleichgerichteter Änderung der QRS-relativen T-Amplitude einher; Effekte unterhalb 0,010 mV je mmol/L (0,1 mm) sind ausgeschlossen; Gruppenaussage, Zeitsatz |
| H1 `positiv_unter_grenze` | gleichgerichteter Zusammenhang nachweisbar; Effekte ab 0,010 mV je mmol/L ausgeschlossen |
| H1 `positiv_grenze_offen` | gleichgerichtet nachweisbar; Erreichen der Grenze offen |
| H1 `grenze_ausgeschlossen` | kein Nachweis; Effekte ab der Grenze ausgeschlossen |
| H1 `unpraezise` | weder Zusammenhang noch Effekt ab der Grenze ausgeschlossen |
| H1 `negativ` | Zusammenhang entgegen der Erwartung; nach technischer Prüfung (Polaritätsgruppen, Kontrollen) interpretierbar |
| H2, H3 | Kategorien wie H1 (H3 gespiegelt); H3 nur Gruppenmittel, Bikarbonat ist Surrogat |
| H4 `beschrieben` | Unterscheidungsmaße zwischen PE und RN gegen diese Gruppendefinition; beschreibend, kein Entscheidungs- oder Diagnoseanspruch, kein Pseudohyperkaliämie-Anspruch |
| Klinische Übersetzung | mm je mmol/L; Wiederholungsgrenze; individuelle Änderung gemessen an RC; Gruppenaussage, keine individuelle Nachweisgrenze, keine Aussage über visuelle Erkennbarkeit |
| Alles | Keine "first"- oder "novel"-Aussage ohne Neuheitsprüfung; jede Aussage trägt den Zeitsatz (4.1) |

## 10. Optionale kaliumblinde Referenzannotation (Frage an den Anwender)

**Ob** und **wann** (vor dem Gesamtlauf) entscheidet der Anwender. Zweck: Prüfung der Messrichtigkeit der Pipeline (Zweitmeinung: Wiederholbarkeit beweist keine Richtigkeit). Ablauf, wenn der Anwender zustimmt:
1. Nach der Bindung `annotation`: Zufallsstichprobe von rund 100 Blöcken (eine Entnahme je Person, **nur Train und Val**, unabhängig von Kalium, Behandlung und Messergebnis; der Testsplit bleibt zu). Bilder (10-s-Streifen, Raster 25 mm/s und 10 mm/mV) mit zufälliger Kennung; **Liste ohne Kalium, ohne Behandlung, ohne Pipelinewerte**; Schlüsseltabelle privat.
2. Der Anwender misst je Bild die T-Amplitude (Höhe des T-Gipfels über der PR-Strecke in mm; positiv, wenn T in der überwiegenden QRS-Richtung liegt) **ohne Kenntnis des Kaliums**.
3. `annotation_auswertung`: Bias, Differenz-SD, Grenzen der Übereinstimmung, ICC(1,1), Spearman, Anteil innerhalb 0,05 mV. **Beschreibend, ohne Tor**; ob eine schlechte Übereinstimmung zu einer Änderung der Messkette (neue Bindung) führt, entscheidet die Koordination.
Der Anwender sieht keine Kaliumwerte; seine Messung geht nur in diese Beschreibung ein.

## 11. Skript, Bindung, Tests

**Skript** `s3_auswertung_v2_2.py` (nutzt die feste Modulliste `s3_signal.py`, `n3/aw197_messung.py`, `aw203_messung.py`, unverändert). Modi: `selbsttest`, `sperren` (die Koordination bindet: SHA-256 von Spezifikation, Skript, den festen Modulen, Konfiguration, Vertrag), `gesamtlauf [--arbeiter N]`, `annotation`. **Start-Prüfung** (`pruefe_start(lock, cfg, spec_path)`) hasht die tatsächlich vorhandenen Dateien (feste Modulliste) und die übergebene `cfg` und verlangt Übereinstimmung mit der Bindung `spec_lock_s3_v2_2.json`; eine austauschbare Hashfunktion oder Modulliste gibt es nicht (Parameter entfernt, per Test geprüft); die `Freigabe` trägt die gebundenen Modulhashes, die jeder Modulzugriff zur Laufzeit prüft (`setze_bindung`, `module_laden`); sie stellt eine `Freigabe` aus, die nur auf diesem Weg entsteht (`Freigabe(...)` von außen wirft einen Fehler). `split_csns`, `alle_csns`, `laden`, `entnahmen_tabelle`, `extrahieren`, `tabellen_bauen` verlangen die Freigabe; `laden` akzeptiert nur Besuchsmengen aus `split_csns` und hängt den Freigabe-Hash an das Protokoll `meta` an (überschreibt nie); die weiteren Funktionen prüfen, dass der letzte Eintrag der aktuellen Freigabe entspricht, und `tabellen_bauen` verweigert Bestände mit Messungen anderer Bindung.

**Selbsttest an Kunstdaten: 40 von 40 bestanden.** Gezielte Tests der Runde 3: (1) `chain(cfg)` überträgt `min_cover_s`, `hf_hp` (mit abgeleitetem Rauschfilter) und `aw203_v2.min_r` aus zwei verschiedenen `cfg` und liefert bei Rückkehr zur ersten dieselbe Messung; eine Manipulation des Modulzustands wird korrigiert; (2) die Modulliste ist gehasht und importiert zugleich (`__file__`-Vergleich), `aw203` nutzt das gehashte `aw197`, falscher Hash bricht ab, im frischen Prozesszustand **vor** der Ausführung, ohne Bindung gibt es keinen Messkettenzugriff, die Signaturen enthalten keine `abhaengig`-Liste; (3) Wiederaufnahme mit gleicher Bindung verwendet vorhandene Messungen, mit neuer Bindung werden sie archiviert (Zählung, Zeitstempel, Bindungs-Hash) und neu gemessen, `laden` hängt an (`meta` zwei Einträge, erster unverändert), `tabellen_bauen` verweigert fremde Messungen und alte Freigaben, Tabellen ohne Bindungsspalte (altes Format) werden archiviert. Gezielte Tests je Muss-Punkt der Runde 2: (1) Parametersignaturen ohne `hash_fn`, jede Hash-Änderung (Spezifikation, Skript, Messkettendatei, Konfiguration, Vertrag, Pipeline) und abweichende `cfg` brechen ab, fehlende oder falsche `SHA256SUMS.txt` bricht vor dem Laden ab (keine Datenbank entsteht); (2) Serum mit späterem Ergebnis neben POC mit frühem Ergebnis, fehlende Serumzeit schließt das Paar aus; (3) Fenster [0, 60] und [45, 105] bei Mittenabstand 90 s, Rohamplituden-Sensitivität ohne Kollisionspaare, Wiederholbarkeit ohne überlappende Fenster, im Ende-zu-Ende-Lauf keine überlappenden Wiederholungsfenster; (4) explizite Zählfeldliste (u. a. `pol_pp`, `dokumentiert`, `rep_fenster_h` unmaskiert) und Vollständigkeitsprüfung aller ganzzahligen Ausgabefelder; (5) die Regression wird in jeder Bootstrap-Ziehung neu geschätzt (Zählung der Schätzungen). Allgemein deckt er ab: Maske nach Zählfeldern (auch Gleitkomma, Seeds unberührt) und vergröberte Zählkette; Behandlungsklassen; Behandlungskategorien mit Rändern (≥, ≤), Calcium, Dialyse, Gabe ohne Zeit, ECG-Zeiten; Polarität; Messkette v3 (negative QRS-Richtung, rohe gegen relative Amplitude); D3-Grenzen (Obergrenze 3 h, Ergebnis vor Auftrag, ohne Ergebniszeit kein D3); Segmentgrenzen; Einheits- und Zeitvertrag (mV, uV abgelehnt, WFDB-Kopf als UTC, CSV mit Z als UTC, Zeitprüfung, Eingabehashes); globale Blocksuche über Segmente (D1 unabhängig, Wiederholung über Segmente, gestörte Mitte); NK1-Blockprüfung; Paarbildung ohne Ergänzung; Serum-Zuordnung bei Doppelwerten, Zensur, Einheiten, H3 nur Serum; Primärkette mit allen Ausschlüssen; Spline-Differenz und identische Fälle (HF-gekoppelte Verzerrung sichtbar); H1 mit Drift, Ausreißern, Huber, Einfluss, Bereichen, rohen Amplituden; Kategorien (Gitter, überschneidungsfrei) und Nullnähe; gemischtes Modell (Konvergenz, Vorzeichen, Protokoll); Messfehler und Präzisionsformel; klinische Größen; keine Tore; H3; H4 (erste Wiederholung, Gruppen, Marke); NK1, NK2 (feste Vollstichprobe, Bijektion, nie die eigene Person), Natrium (rein, gekoppelt, Artefakt); Stabilität je Split, D1 gegen D3, alle Quellen; Deutung (auffällige Kontrolle sperrt H1 und H2); Freigabe und Bindung (jeder Hash, kein Datenzugriff ohne Freigabe, auch nicht auf Train/Val); Block-Kollision und Timing; Annotation; **Ende-zu-Ende** mit 40 synthetischen Besuchen (CSV und WFDB, alle drei Splits, Dialyse, falsche Kaliumeinheit, fehlende Ergebniszeit, negative Polarität, Altinsulin): Gesamtlauf nur nach Bindung, Testbesuche im Bestand, maskierte Ausgabe ohne Kennung. Kein Lauf auf MC-MED-Daten.

| Datei | SHA-256 |
|---|---|
| `s3_auswertung_v2_2.py` (Stand des Selbsttests 40 von 40) | 04ac15dae0f8a6a0a9b2d154e43b4ff3d240acd410d10707498fa53d48c9dcea |
| `s3_signal.py` | 6450c7e5d0a0205f4f2333b93c992354bb2df41f84f2be811ca1165fe5784832 |
| `n3/aw197_messung.py` | 03e2b2d28aa04a231c6d3280376255e329fdf4d8fa28465f0b4891ccb35e7f67 |
| `n3/aw203_messung.py` | 37e3b06a9dd84ba4ab2f16d98c940ad04d42325a8348d12ed21bf04e0614f193 |

## 12. Setzungen und offene Punkte für die Koordination

| Nr. | Punkt | Stand |
|---|---|---|
| S1 | **Entscheidung Nr. 10 gegen Nr. 8:** "gesamt sperrt Urteile bei Messtor-Versagen" widerspricht "PK und Wiederholbarkeit sind keine Tore". Umgesetzt: keine Tore; Urteile sind nur gesperrt, wenn die Primärmenge nicht schätzbar ist; eine auffällige biologische Kontrolle macht H1/H2 nicht deutbar. | Bestätigung durch die Koordination |
| S2 | Primär = kein Hinweis und Erstwert < 6,0; "Abfall durch Behandlung erklärt" wirkt nur über das Behandlungsstratum (dokumentiert), nicht im Primärmodell. | Bestätigung durch die Koordination |
| S3 | Gesamtanalyse statt Bestätigung: Splits sind besuchsweise zufällig, nicht personenunabhängig; Stabilität je Split ist deshalb nur Beschreibung. | zur Kenntnis |
| S4 | Vorläufe der Behandlungsfenster 120 min (dokumentiert) und 360 min (unklar), Dialyse über die Auftragszeit (Beginn nicht dokumentiert), Calcium als ECG-wirksam. | Bestätigung durch die Koordination |
| S5 | HF-Spline mit festen Knoten 55, 70, 85, 110/min; Marke T ≥ 0,30 mV (3 mm); AUC-Marke 0,65; Äquivalenzgrenzen 0,005. | Bestätigung durch die Koordination |
| S6 | Grenze H2 0,2 mV/s je mmol/L; Grenze H3 0,29 ms je mmol/L. | Bestätigung durch die Koordination |
| S7 | H4-Wiederholungsfenster 0,25 bis 6 h, Grauzone 5,0 bis 5,5, Abfall ≥ 1,0. | Bestätigung durch die Koordination |
| S8 | **Präzision:** σ_res nicht ohne Zielgrößensicht schätzbar; Planung deckt 0,06 bis 0,127 mV ab; erwartete Primärpaare 200 bis 330. Bei den ungünstigen Werten ist `positiv_ueber_grenze` unwahrscheinlich. | zur Kenntnis; Koordination: Grenze G = 0,010 beibehalten? |
| S9 | NK1 hat bei geringer ECG-Abdeckung (versetzte Fenster 2 bis 6 h) wahrscheinlich nur wenige Dutzend Paare; Halbbreite wird berichtet. | zur Kenntnis |
| S10 | Wellenform-Hashes (Ableitung II) wie S5 vor der Bindung prüfen. | offen |
| S11 | Neuheitsprüfung (Literatur seriell Kalium und Einzelableitungs-T-Welle); Zusammenführung beider Fragestellungen in der Salami-Prüfung. | offen |
| S12 | Der Anwender entscheidet über die optionale Annotation (Abschnitt 10) und kann im Rahmen der Bindung die Reihenfolge Annotation vor Gesamtlauf festlegen. | Frage an den Anwender |

## 13. Abhakliste

### Zweitmeinung zu v1

| Punkt | Umsetzung |
|---|---|
| D3 pragmatisch; ECG-Abstände zu beiden Zeitstempeln; identische/überlappende Fenster; D1 gegen D3 auf denselben Paaren | 4.3; `timing_beschreibung`, `kollision`, `d1_vergleich` |
| "Unbehandelt" zu stark; drei Kategorien; Fenster über Zuordnungen und ECG; Calcium, Dialyse | 4.2; `behandlung_kategorie` |
| Regression zur Mitte, Ausgangsbereiche, Anstiege/Abfälle, Messfehler | 6.1; `ausgangsbereich`, `messfehler_szenarien` |
| Freier Achsenabschnitt; OLS nicht robust | 6.1; Huber, Einfluss |
| HF: unadjustiert und adjustiert auf identischen Fällen, Spline f(HF₂) − f(HF₁) | 6.1; `h1_modelle`, `spline_differenz` |
| Polarität: QRS-relativ, Gruppen sichtbar, rohe Amplitude | 5; Sensitivitäten (a/b) |
| BH/VP sind Verlaufsgruppen; Umbenennung; gleiche Wiederholungsfenster; unklare Gruppe; Erstkalium und Abstand; Begründung der Untergrenze | 6.4 |
| Pseudohyperkaliämie nur mögliche Erklärung; i-STAT nicht pauschal hämolysefrei | 6.4 |
| Gesamtschätzung statt "konsistent_unpräzise" als Bestätigung; Machbarkeitssicht offenlegen; Split nur Stabilität | 0, 3.1, 8 |
| 300-Paare-Tor unbegründet; Präzision aus ΔK-Streuung, Residual, Clustern | 8 |
| PK kein Tor; Wiederholbarkeit keine Richtigkeit; kaliumblinde Referenz | 5, 10 |
| NK1 ohne Mindestzahl 100; Natrium keine Nullexposition; NK2 Pipeline; "enthält null" ist keine Nullnähe | 7 |
| Überlappende Kategorien; "klein ausgeschlossen" umbenannt; negative Zusammenhänge; H3 Bikarbonat | 6.1, 6.3, 9 |
| Klinisch: Wiederholungsgrenzen, individuelle Streuung, 0,1/β entfällt, 3 mm/AUC 0,65 nur Setzungen | 2 |
| Selektion, Zensur, einflussreiche Paare, gemeinsame Fälle | 4.1, 4.4, 6.1 |
| Publikationskonzept an MC-MED | (siehe Publikationskonzept des Autors) |

### Code-Review zu `s3_auswertung_v1.py`

| Befund | Behebung in v2 |
|---|---|
| 1 Testsperre intern unzureichend; `laden` ohne Prüfung | `Freigabe` nur aus `pruefe_start`; `pflicht_freigabe` in `split_csns`, `laden`, `entnahmen_tabelle`, `extrahieren`, `tabellen_bauen`; `CSNMenge`; Freigabe-Hash im Bestand |
| 2 Freeze unvollständig | Freeze entfällt (eine Gesamtanalyse); Bindung prüft Hashes von Spezifikation, Skript, Messkette, Konfiguration, Vertrag |
| 3 Tore und Sperren in `gesamt`; Entwicklung/Test nicht getrennt | keine Tore; eine Auswertung; Sperre nur bei fehlender Schätzbarkeit; auffällige Kontrolle macht H1/H2 nicht deutbar |
| 4 D3-Obergrenze, Segmentgrenzen, 60-Minuten-Ergänzung | `d3_grenzen`, `messe_bei`, `messfenster_um` an Segmentgrenzen, ohne Ergebniszeit Ausschluss |
| 5 D1 nur im D3-Cache; Wiederholung nur im selben Segment; gleicher Block an beiden Enden | `naechster_block` global, Wiederholung über Segmente in absoluter Zeit, `kollision` |
| 6 Behandlungsrand `>` | `>=` und `<=`, Test mit Rändern |
| 7 MixedLM-Prüfung, Warnungen, H2 in Deutung | `gemischtes_modell` (Konvergenz, Vorzeichen, Protokoll); H2 in `deutung` |
| 8 NK1 prüft Zielzeit; Polarität versetzter Fenster; NK2 unterschiedliche Teilstichproben | `shift_gueltig` am tatsächlichen Block; NK1 verlangt sichere gleiche Polarität; NK2 feste Vollstichprobe |
| 9 `bool_and`-Serumverlust; H3 ohne Serumfilter; Zeit/Einheiten | `k_s`/`k_o` getrennt; H3 nur Serum; `VERTRAG`, `einheit_pruefen`, `zeitvertrag_pruefen` |
| Maske pauschal | `mask_tree` nach Zählfeldern, `zaehlkette_vergroebern` |

### Code-Review zu `s3_auswertung_v2.py` (fünf Muss-Punkte)

| Muss-Punkt | Behebung in v2.1 (gilt unverändert; Signaturen siehe 0b) |
|---|---|
| 1 Bindung und Eingabeintegrität | `hash_fn` entfernt; `hashes(cfg, spec_path, abhaengig)` hasht die tatsächlich verwendeten Dateien und die übergebene `cfg`; `schreibe_lock` und `pruefe_start` verwenden dieselbe Funktion; `eingaben_pruefen` bricht ab, wenn `SHA256SUMS.txt` fehlt oder ein Hash abweicht, vor dem Laden |
| 2 Ergebniszeiten quellengetreu | `r_s`/`r_o` getrennt aggregiert; `draws_serum` und `draws_alle` übernehmen die Zeit der eigenen Quelle; fehlende Serumzeit gibt ein Paar `ohne_ergebnis` |
| 3 Überlappung überall | `~Q.kollision` auch in der Rohamplituden-Sensitivität; `naechster_block(..., fenster_ausser=...)` prüft das tatsächliche Fenster; `wh_beschreibung` schließt überlappende Fensterpaare aus und zählt sie |
| 4 Maske | `ZAEHLFELDER` (explizit), Vollständigkeitstest über alle ganzzahligen Ausgabefelder |
| 5 H4-AUC | adjustierende Regression innerhalb jeder Bootstrap-Ziehung (`fr(t)` schätzt neu) |

### Code-Review zu `s3_auswertung_v2_1.py` (drei Bindungshindernisse)

| Hindernis | Behebung in v2.2 |
|---|---|
| 1 Verifizierte und verwendete Konfiguration | `chain(cfg)` mit allen Messparametern aus der gebundenen `cfg`, Übertragung und Prüfung, Neuübertragung bei Wechsel; Rauschfilter wird aus `hf_hp` neu abgeleitet |
| 2 Austauschbare Abhängigkeiten | feste Modulliste, Hash und Import über dieselben Pfade (ausgeführt werden die gehashten Bytes), Laufzeitprüfung gegen die Bindung, auch in Workern; Parameter `abhaengig` entfernt |
| 3 Wiederaufnahme umgeht Korrekturen | Bindungs-Hash in `mess_v2` und `mess_done`; fremde Messungen in Archivtabellen verschoben und neu gemessen; `meta` als Protokoll, `laden` überschreibt nicht |

### Entscheidungen der Koordination

Nr. 1 bis 10 siehe oben; Nr. 11 Abschnitt 10; Nr. 12 Publikationskonzept des Autors; klinische Zielgröße Abschnitt 2.

## 14. Bekanntes (nicht bindungshindernd, Entscheidung der Koordination nach Runde 4 der Code-Review)

| Nr. | Punkt | Stand |
|---|---|---|
| B1 | **Robustheitslücke in `chain(cfg)` (Z. 557):** Bei unveränderter Konfiguration erkennt die Bedingung nachträgliche Änderungen an `aw197.CFG`, aber **keine alleinige Änderung an `aw203.V2`**. Im regulären Ablauf gibt es dafür keinen Auslöser: Die Module werden nur über `chain(cfg)` konfiguriert, kein anderer Code schreibt in `V2`, ein Wechsel der Konfiguration wird vollständig übertragen und geprüft, und die Modulbytes sind zur Laufzeit gegen die Bindung gehasht. Bewusst unverändert gelassen (eine Änderung würde Skript-Hash und Review ungültig machen). | bekannt, akzeptiert |
| B2 | `annotation_vorbereiten` extrahiert alle Kandidaten von Train und Val mit einem Prozess und ist für die Annotation sehr langsam. Der Ablauf nutzt dafür einen ungebundenen Treiberpfad (`s3_gesamtlauf_treiber.py annotation`), der dieselben gebundenen Funktionen (Laden, Messung, Stichprobe, Bild) auf einer zufälligen, kaliumunabhängigen Teilmenge der Entnahmen von Train und Val ausführt. Er ändert weder Skript noch Konfiguration noch die Gesamtauswertung. | bekannt, akzeptiert |
