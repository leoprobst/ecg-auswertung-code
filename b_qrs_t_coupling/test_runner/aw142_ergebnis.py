"""AW142: vollständiger Bericht ausschließlich aus finalen Runneraggregaten.

Keine Fits, keine Personen-/Signalinputs, keine Änderung gebundener Dateien.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(str(data_path("b.aw142")))
RUN = Path(str(data_path("b.aw136_b_testlauf_aw142")))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def js(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def fmt(x):
    if x is None or x == '':
        return '-'
    if isinstance(x, float):
        return f'{x:.8g}'
    return str(x).replace('|', '/').replace('\n', ' ')


def main():
    source = RUN/'ergebnis.json'
    assert source.stat().st_size < 20*1024*1024
    result = json.loads(source.read_text(encoding='utf8'))
    assert result['role'] == 'test' and result['partition'] == 'test'
    proof = json.loads((OUT/'vorpruefung.json').read_text(encoding='utf8'))
    assert all(c['ok'] and sha(c['path']) == c['expected'] for c in proof['checks'])
    assert all(c['ok'] for c in proof['hashes_after_tests'])
    assert set(result['input_stops']) == {'BEF-0042', 'BEF-0081', 'BEF-0075'}
    release = json.loads((PROJECT/'werkzeuge/b_test/FREIGABE_K.json').read_text(encoding='utf8'))
    for key in ('runner_sha256', 'parameter_sha256', 'spec_sha256'):
        assert result[key] == release[key]
    manifest = json.loads(Path(release['input_manifest_path']).read_text(encoding='utf8'))
    package = json.loads((PROJECT/'werkzeuge/b_test/parameter_entwicklung.json').read_text(encoding='utf8'))
    rows = []
    details = []

    def emit(label, key, value, bef, primary=False):
        boot = value.get('bootstrap', {})
        ci = value.get('or_ci95_wald') or [None, None]
        dci = boot.get('ci95', {}).get('delta_auc') or [None, None]
        oci = boot.get('ci95', {}).get('or') or [None, None]
        counts = value.get('class_persons', {})
        mult = value.get('multiplicity', {})
        reason = value.get('reason', '')
        if label.startswith(('3 ', '3b ')):
            reason = 'nicht schätzbar (Runnerlogik bei Endpunktstopps); Runnergrund: '+reason
        descriptive = not primary or bef in ('BEF-0075', 'BEF-0087')
        success = value.get('urteil', 'nicht_schaetzbar')
        if descriptive and value.get('status') == 'schaetzbar':
            success = 'deskriptiv; lokale Runnerregel='+success
        row = dict(bef=bef, part=label, status=value.get('status', ''), reason=reason,
                   n=value.get('N'), persons=value.get('G'), cases=counts.get('1'), controls=counts.get('0'),
                   or_per_sd=value.get('or'), ci95_lower=ci[0], ci95_upper=ci[1], p_raw=value.get('p_raw'),
                   p_holm=mult.get('p_holm') if value.get('p_raw') is not None else None,
                   hierarchy=js(mult) if primary else 'deskriptive Sensitivität; kein Ersatz des Haupttests',
                   success=success, beta=value.get('beta'), beta_ci95_wald=js(value.get('beta_ci95_wald')),
                   auc_G=value.get('auc_G'), auc_G_marker=value.get('auc_G_marker'),
                   auc_marker_original_steigend=value.get('auc_marker_original_steigend'),
                   delta_auc=value.get('delta_auc'), delta_auc_ci95_lower=dci[0], delta_auc_ci95_upper=dci[1],
                   bootstrap_planned=boot.get('planned'), bootstrap_valid=boot.get('valid'),
                   bootstrap_status=boot.get('status'), bootstrap_failed=js(boot.get('failed', {})),
                   or_bootstrap_ci95_lower=oci[0], or_bootstrap_ci95_upper=oci[1],
                   missingness_fallfluss=js(value.get('missingness_fallfluss', {})),
                   fallen=js(value.get('fallen', {})), blinding='Analyst nicht verblindet' if bef=='BEF-0087' else 'siehe Offenlegungen K',
                   source_run='A155-2026-10-04-d6e70e; AW138-Testprojektionen',
                   source_file=str(source), source_sha256=sha(source), source_key=key,
                   script='werkzeuge/b_test/runner.py -> adapter.py/statistik.py; Bericht aw142_ergebnis.py')
        rows.append(row)
        details.append((label, key, value))

    for bef, value in result['results'].items():
        emit('B3 Hauptmodell', 'results/'+bef, value, bef, True)
    for i, value in enumerate(result['sensitivities']):
        label = value['rank']+' '+value['analysis']
        match = re.search(r'BEF-\d+', value['analysis'])
        bef = match.group() if match else 'BEF-0087' if value['rank'] in ('5a', '5b', '6e_RV') else 'BEF-0063'
        key = f'sensitivities/{i}'
        if 'B4' in value:
            for sub in ('B4', 'B3_same_set'):
                emit(label+' / '+sub, key+'/'+sub, value[sub], bef)
            details.append((label+' / gepaarter Nenner', key, {k:v for k,v in value.items() if k not in ('B4','B3_same_set')}))
        else:
            emit(label, key, value, bef)
    csvpath = PROJECT/'nachrechnung/b_testergebnis.csv'
    with csvpath.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def table(selected):
        header = '| Teil | N / Fälle / Kontrollen | OR je Entwicklungs-SD [Wald-95%-KI] | p roh / Holm | ΔAUC [Bootstrap-95%-KI] | gültiger Bootstrap | Status / Entscheidung / Grund | Quelle/Schlüssel |\n|---|---|---|---|---|---|---|---|'
        lines = [header]
        for row in selected:
            lines.append('| '+' | '.join([
                row['bef']+' / '+row['part'],
                ' / '.join(fmt(row[k]) for k in ('n','cases','controls')),
                fmt(row['or_per_sd'])+' ['+fmt(row['ci95_lower'])+'; '+fmt(row['ci95_upper'])+']',
                fmt(row['p_raw'])+' / '+fmt(row['p_holm']),
                fmt(row['delta_auc'])+' ['+fmt(row['delta_auc_ci95_lower'])+'; '+fmt(row['delta_auc_ci95_upper'])+']',
                fmt(row['bootstrap_valid']),
                fmt(row['status']+'; '+row['success']+'; '+row['reason']),
                '`ergebnis.json/'+row['source_key']+'`'])+' |')
        return '\n'.join(lines)

    suites = list(ET.parse(OUT/'tests/junit.xml').getroot().iter('testsuite'))
    teststats = {k:sum(int(s.get(k, 0)) for s in suites) for k in ('tests','failures','errors')}
    stamp = datetime.now(timezone.utc).isoformat()
    h1 = result['results']['BEF-0063']
    h4 = result['results']['BEF-0049']
    gate = (PROJECT/'nachrechnung/B_MESSGATE_ABNAHME_K.md').read_text(encoding='utf8')
    files = [source, OUT/'vorpruefung.json', OUT/'tests/junit.xml', OUT/'tests/resources.json',
             PROJECT/'werkzeuge/b_test/FREIGABE_K.json', Path(release['input_manifest_path']),
             PROJECT/'werkzeuge/b_test/aw142_vorpruefung.py', PROJECT/'werkzeuge/b_test/aw142_start.ps1',
             Path(__file__), csvpath]
    hashes = '\n'.join('| `'+str(p)+'` | `'+sha(p)+'` |' for p in files)
    checks = '\n'.join('| `'+c['path']+'` | `'+c['actual']+'` | bestanden |' for c in proof['checks'])
    flowtext = []
    for bef, value in result['results'].items():
        sourceentry = manifest['hypotheses'].get(bef, {})
        flowtext.append('### '+bef+'\n\nQuelle: `ergebnis.json/results/'+bef+'`; Projektionsvertrag `input_manifest_v5.json/hypotheses/'+bef+'`.\n\n'
                        +'Projektionspfad und Bindung: `'+str(sourceentry.get('path'))+'`, SHA `'+str(sourceentry.get('sha256'))+'`. '
                        +'Der Runner stoppt gesperrte Teile vor dem Inputlesen. Kandidatenpool vor `select_index` wird nicht als Fallflusszahl exportiert; kein zusätzlicher Patientenzugriff für diese Berichtslücke.\n\n```json\n'
                        +json.dumps(value.get('missingness_fallfluss', {'nicht_schaetzbar':value.get('reason')}), ensure_ascii=False, indent=2)+'\n```')
    fulltext = '\n\n'.join('### '+label+'\n\nQuelle: `'+str(source)+'`, Schlüssel `'+key+'`, Skript gebundener Runner/Adapter.\n\n```json\n'
                             +json.dumps(value, ensure_ascii=False, indent=2)+'\n```' for label,key,value in details)
    report = f'''# AW-142 - einmaliger B-Testlauf: Ergebnisse und vollständige Kontrollspur

Stand {stamp} · MS/PR · Rechenspur R · AGENTS.md vollständig gelesen · keine Recherche.

## Ergebnis

**Der freigegebene einmalige B-Testlauf ist abgeschlossen. H1: {h1['urteil']}; H4: {h4['urteil']}.** Das H1-Gate ist `{result['H1_gate']}`. H2, H3 und BEF-0075 sind nicht schätzbar; die sekundäre Holm-Familie bleibt unverändert H2-H4 mit drei Tests. BEF-0087 ist ausschließlich deskriptiv: **Analyst nicht verblindet**. Die Zahlen beschreiben K1-Assoziationsneufits in der Testpopulation; die Modell-AUC sind ausdrücklich **apparente AUC im Assoziationsfit**, keine gehaltene Vorhersageleistung. Keine klinische Gradvergabe oder Manuskriptbewertung durch diesen Auftrag.

{table(rows[:6])}

Quelle sämtlicher Tabellenzahlen: `{source}`, SHA `{sha(source)}`, jeweiliger Schlüssel in der letzten Spalte; Feldzuordnung: N/G, class_persons/1 bzw. /0, or, or_ci95_wald, p_raw, multiplicity/p_holm, delta_auc, bootstrap/ci95/delta_auc, bootstrap/valid. Leere Felder = nicht schätzbar/nicht exportiert, keine Nullwerte. CSV enthält ungerundete Werte, vollständige Fallflüsse, Bootstrapfehler und F1-F8 je Ergebnis. Intervallgrenzen der OR in der Tabelle sind CR1-Wald-KI; deskriptive Bootstrap-OR-KI stehen separat im Detailanhang/CSV.

## Bindende Regeln und Hierarchie

Freeze v4 **§0 hat Vorrang**: H1 primär LVEF **<40 % gegen ≥40 %**; inklusive ≤40 % gegen >40 % ist Rang 2b. B3 einziges konfirmatorisches Grundmodell; B4 ist Sensitivität. Echozeitflag entfällt für H1/RV gemäß §0 Nachtrag 1, Echo-ID/Timestamp/±30-Tage-/Konfliktregeln bleiben. Geschlechtsstrata entfernen ausschließlich sex_numeric vor Fit gemäß §0 Nachtrag 2(a). Keine Änderung der gebundenen Dateien oder der Modelle.

H1 wird durch Richtung β>0, zweiseitiges p≤0,05 und untere Wald-β-KI>0 entschieden (§7). H4 nur bei offenem H1-Gate, Holm-Verwerfung in der unverkleinerten Dreierfamilie und festem positiven Vorzeichen. Nichtschätzbare H2/H3 behalten intern p=1; berichteter p bleibt nicht schätzbar. Vollständige Runner-Hierarchie: `ergebnis.json/results/BEF-0042,multiplicity`, analog BEF-0081/0049; im Detailanhang. Kontrollen erhalten keine konfirmatorischen Erfolge. Ein lokales `urteil=erfolg` des Runners für BEF-0087 oder eine Sensitivität wird nur als Erfüllung der lokalen numerischen Regel ausgewiesen, niemals als konfirmatorischer Claim. K1-Erfolg ist keine automatische E3-Promotion.

Die logistischen Neufits verwenden feste Entwicklungs-Skalierungen, Ridge 1e−6, Patientencluster subject_id und CR1-Waldinferenz. Deskriptiver Personenbootstrap: 1000 geplante Replikate, Seed 1282026, gepaarte Multiplizitäten über die feste Personenunion; keine Nachziehung. Mindestens 950 gültige Replikate für ein Bootstrap-KI. Sonst bleibt dessen Intervall nicht schätzbar, unabhängig vom Wald-Haupttest. Quelle numerischer Gates: Freeze §§6-8 und ergebnis.json/je Fit; Solver-/Rang-/Separation-/CR1-Felder im Anhang.

## B4 und sämtliche implementierten Sensitivitäten

{table(rows[6:])}

**IPW: nicht schätzbar (Runnerlogik bei Endpunktstopps).** Gleiches gilt für den separaten Verifikationsfit. `runner.py::test_analysis` lädt technical/verification_train nur bei `not stops`; die vorgesehenen H2/H3/BEF-0075-Stopps verhindern das. Rohgrund im Ergebnis: fehlendes erstes technisches Indexuniversum. Keine ESS-/Propensity-/Gewichtquantile, keine gewichtete OR oder IPW-Richtungsprüfung verfügbar; keine Ersatzgewichte und keine Runneränderung. Dies wurde vor dem Lauf von K im Auftragsnachtrag ausdrücklich akzeptiert. F7 bleibt offen.

Signalränge 7/8a/8b/9: **nicht schätzbar (Skalierung nicht gebunden)** gemäß Freeze §0 Nachtrag 2(b); Pilot ist kein Ersatz. EchoNext ausgeschlossen. LSB-LVH bleibt nicht schätzbarer Entwicklungs-Supplementbefund. NT-Zusatzpfade werden mit dem vom Runner gelieferten Fehlergrund vollständig ausgewiesen; der knappe KeyError-Grund `names` verweist auf ungebundene NT-Designparameter und ist kein biologischer Nullbefund. Keine Nachholung oder günstiger Ersatz nach Ergebnissicht. Der Störtest nach SPEZIFIKATION_B_STOERTEST_v1 ist eine gesonderte nachgelagerte Rechnung; AW-142 führt die im gebundenen B-Runner implementierten Teile aus und erteilt keinen zusätzlichen Signalrechenauftrag.

## Fallfluss, Nenner und Berichtslücken

Je Hauptteil unten: sequentielle Tore, überlappende Ausschlusszahlen, Missingness je Designspalte und Strata nach Referenzklasse/Alter/Geschlecht/Split. Die Klasse in den Missingness-Strata stammt aus label_state; bei inklusiver LVEF-Sensitivität ersetzt sie nicht die Neuklassifikation durch den Fit. Finale Fälle/Kontrollen stammen ausschließlich aus class_persons. Personen und Zeilen stehen als G/N getrennt im Anhang. Der Runner-Fallfluss der Untergruppen beschreibt zunächst die Basismasken; zusätzliche Selektoren werden danach angewandt. Daher sind dort complete_B3_marker und n_selected/N nicht austauschbar. B4 verwendet ebenfalls den technischen Feldnamen complete_B3_marker, obwohl strong=True die B4-Matrix prüft. Diese Dokumentationsnamen ändern die ausgeführten Gates nicht.

Der vollständige Kandidatenpool-zu-Index-Fluss vor select_index wird nicht ausgegeben. Nichtschätzbare Fits exportieren teils keine Klassenhäufigkeiten bzw. kein N; die vorhandenen n_selected/Masken werden im Anhang erhalten. Keine zusätzlichen Testdatenlesungen zum Auffüllen dieser Lücken, keine erfundenen Fälle/Kontrollen. Diese Grenzen sind als offene Berichtspunkte an K abgegeben.

{chr(10).join(flowtext)}

## Fallenprüfung F1-F8 - je Hauptteil und Kontrolle

Jede Ergebniszeile enthält ihre eigene `fallen`-Map (CSV und Detailanhang), einschließlich Null-/Gegen-/Nichtschätzbarkeitsbefunden. Die folgenden Begründungen gelten für jede dieser Zeilen; endpoint-/teilbezogene Besonderheiten stehen in Grund/Fallfluss/Hierarchie. Kein Status wird durch statistischen Erfolg aufgewertet.

| Falle | Status | Begründung / konkrete Grenze |
|---|---|---|
| F1 | eingeschraenkt | Echo/ICD-Referenz; geometrische/Orientierungskopplung begrenzt reinen Formclaim. ICD-Kardiomyopathie ist Referenzproxy, keine Adjudikation. |
| F2 | eingeschraenkt | Verifizierte Krankenhaus-Complete-cases, keine Populationsprävalenz/PPV/NPV. |
| F3 | nicht_zutreffend | K1 mit festen Referenzgrenzen, kein optimierter diagnostischer Arbeitspunkt. |
| F4 | eingeschraenkt | Entwicklungswahl bleibt post hoc; feste H1→Holm-H2-H4-Hierarchie. BEF-0087 Analyst nicht verblindet und deskriptiv; Offenlegungen K unten. |
| F5 | offen | Informative Referenz-/Signal-/Kovariatenmissingness; vollständige Fallflussaggregate, keine Imputation oder Ersatz-EKG-Wahl. |
| F6 | eingeschraenkt | B3 mit obligatem Integralwinkel, B4 gepaart auf engerem Fallset; ungleiche Wiederholbarkeit bleibt Grenze. Keine globale Optimalität/Confoundingfreiheit. |
| F7 | offen | Echoindikation/Verifikation, IPW nicht schätzbar (Runnerlogik bei Endpunktstopps); keine MAR-/Innerhalb-Person-Aussage. Stundenendpunkte gestoppt. |
| F8 | offen | Gebundene 500-Hz-Korswerte; Delineation/Filter/Gerät/Pacingrest und ungebundene Signalvarianten begrenzen Transport. Berichtstext kein Prädiktor. |

Widerlegen/begrenzen würden gemäß Freeze ungültige Referenzzeit, gegenläufige Test-OR, fehlender Zusatz auf stärkerer identischer Basis oder nachgewiesene Messartefakte. Nichtschätzbarkeit und erfolglose Untergruppen beweisen keine Äquivalenz und retten/ersetzen keinen Haupttest.

## Durchführung, Tests, Ressourcen und Unverändertheit

Vorprüfung ohne produktive Testinputs: {len(proof['checks'])} Hashvergleiche bestanden, genau die drei erlaubten Endpunktstopps; sämtliche Bindungen auch nach Tests und final vor Berichterstellung unverändert. Neue K-Freigabe/Manifest v5 gemäß Nachtrag. Explizite Testdateiliste: {js(proof['test_files'])}. JUnit: {js(teststats)}, mithin 103 bestanden und der eine von K akzeptierte Fehler in test_vertrag.py::test_splitbarriere_vor_parameter_und_inputzugriff. Er erwartet eine fehlende Freigabedatei, die inzwischen existiert; tatsächlicher FileNotFoundError betrifft eine absichtlich fehlende synthetische Parameterdatei, vor produktivem Datenzugriff. Keine Tests geändert/übersprungen. --noconftest plus gleichartige AW142-Ressourcen-/tmp-Fixtures verhindern AW136-Schreibzugriffe der originalen Fixture.

Aufruf: `powershell -NoProfile -ExecutionPolicy Bypass -File werkzeuge/b_test/aw142_start.ps1 -Phase preflight`. Produktiv ruft der neue Wrapper **unverändert** `werkzeuge/b_test/start.ps1 -Mode runner -Role test -Output <local path>` auf. Freies RAM/Commit im gebundenen Start ≥4 GiB; Gate unverändert, numerische Threads=1, ein Pythonprozess. Vorprüfung Ressourcen: `{js(proof['resource_before'])}`; Testbudget in tests/resources.json. Runner-Ressourcen (Bytes; Quelle ergebnis.json/resources): `{js(result['resources'])}`.

Losgelöster Hidden-Start via Start-Process, eigener Supervisor PID 24736, Status <local path>, Logs run_stdout.log/run_stderr.log, Checkpoints unter {RUN}/checkpoints (alle 25 Replikate gebunden). Der erste Start-Process-Aufruf scheiterte vor Prozessanlage am doppelt vorhandenen Umgebungsschlüssel Path/PATH. Im zweiten Aufruf wurde ausschließlich PATH in der aufrufenden Shell entfernt und powershell.exe absolut adressiert. Keine analytische Änderung, kein bereits gestarteter Pythonlauf/Checkpoint und keine Ergebnisinspektion zu diesem Zeitpunkt; **keine technische Fortsetzung eines Testfits, nur ein erfolgreicher produktiver Start**. Keine Prozesse beendet. Die RAM-Prüfung vor produktivem Pythonstart steht in run_stdout.log. Beginn im Register 18:12:00Z; tatsächlicher Wrapperbeginn status_run.json 18:12:40.8182615Z; Ende ergebnis.json/utc `{result['utc']}`. Frühere STOP-Abgabe 18:09Z bleibt im Register; ihre veraltete Startblockade ist durch K-Nachtrag aufgehoben. aw142_bericht.py ist der erhaltene frühere STOP-Berichtsgenerator und wurde hier nicht erneut ausgeführt.

## Claimgrenze, alle Offenlegungen und Erratum - wörtliche Übernahme

Der folgende vollständige Text ist wortgleich übernommen aus nachrechnung/B_MESSGATE_ABNAHME_K.md, SHA `{sha(PROJECT/'nachrechnung/B_MESSGATE_ABNAHME_K.md')}`. ICC-Zahlen sind Messgategrundlage aus dieser Quelle, keine AW142-Neuberechnung.

{gate}

## Nachweise

| Datei | SHA-256 |
|---|---|
{hashes}

Alle Vorprüfungsbindungen und abschließende Gleichheit (Quelle vorpruefung.json/checks, finale Byteprüfung aw142_ergebnis.py):

| Datei | erwartete = tatsächliche SHA-256 | Status |
|---|---|---|
{checks}

## Vollständige Modellaggregate und P08-Nachweise je Teil

Die folgenden JSON-Blöcke enthalten ausschließlich Aggregate des gebundenen Runners; keine Personenwerte/IDs. Sie bewahren auch Nichtschätzbarkeit, Wald-/Bootstrap-Trennung, Fehlerzählungen, Strukturregeln, Klassen-/Rang-/Separations-/Konvergenz-/CR1-Gates und alle Missingness-Strata unverändert. Der Index i in sensitivities/i ist nullbasiert; Auswahl außerdem über rank und analysis.

{fulltext}

## Offene Fragen, Nebenbefunde und Sicherheit

K bewertet die fachliche Tragweite und Manuskriptverwendung. Nicht exportierte Kandidatenpool-Flüsse bzw. Klassen bei abgebrochenen Fits, knappe NT-Parameterfehler und die IPW-Runnergrenze bleiben offengelegt; keine eigene Reparatur nach Testsicht. Kein zusätzlicher Paketbedarf. Keine externe Literatur/Quellenfehler-Kandidaten geprüft; internes Schlagmedian-Erratum steht oben wörtlich. Sicherheit hoch für unveränderte technische Ausführung und getreue Aggregatübernahme; Interpretationsgrenzen F1-F8 bleiben bestehen. Kein unabhängiger zweiter numerischer Patientenfit (P01b/c) in AW142, weil ein zweiter Lauf nach Ergebnissicht verboten ist; technische Vorbelege aus AW138 sind davon getrennt.
'''
    reportpath = PROJECT/'nachrechnung/B_TESTERGEBNIS_AW142.md'
    reportpath.write_text(report, encoding='utf8')
    # QA der dokumentarischen Übertragung, kein weiterer analytischer Lauf.
    with csvpath.open(encoding='utf-8-sig', newline='') as f:
        exported = list(csv.DictReader(f))
    assert len(exported) == len(rows)
    assert gate in reportpath.read_text(encoding='utf8')
    assert len({r['source_key'] for r in rows}) == len(rows)
    for key in result['results']:
        assert any(r['source_key'] == 'results/'+key for r in rows)
    for i in range(len(result['sensitivities'])):
        assert any(r['source_key'].startswith(f'sensitivities/{i}/') or r['source_key']==f'sensitivities/{i}' for r in rows)
    qa = dict(utc=stamp, result_sha256=sha(source), report_sha256=sha(reportpath), csv_sha256=sha(csvpath),
              rows=len(rows), primary_parts=len(result['results']), sensitivity_entries=len(result['sensitivities']),
              all_bound_hashes_unchanged=True, verbatim_gate=True, csv_complete=True, additional_fits=False)
    (OUT/'bericht_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf8')
    with (PROJECT/'koordination/TESTZUGRIFFE.md').open('a', encoding='utf8') as f:
        f.write(f'\n| {stamp} | AW-142 | Ende: einmaliger Testlauf und Bericht abgeschlossen; eigene Aggregate geprüft | Freeze v4 `{release["spec_sha256"]}`; Manifest v5 `{release["input_manifest_sha256"]}` | Runnerende {result["utc"]}; ergebnis.json SHA `{sha(source)}`; Bericht `{sha(reportpath)}`; CSV `{sha(csvpath)}`. Keine Fortsetzung, kein zweiter Fitlauf, alle gebundenen Hashes unverändert; H1-Gate {result["H1_gate"]}; IPW nicht schätzbar gemäß K. |\n')
    submission = f'''# Abgabe AW-142 - einmaliger B-Testlauf abgeschlossen

Stand {stamp} · MS/PR · Deutsch · AGENTS.md vollständig gelesen.

## Ergebnis

Der Auftrag einschließlich K-Neustartnachtrag ist erledigt: Vorprüfung wiederholt, einmaliger gebundener Testlauf losgelöst durchgeführt, sämtliche Runner-Haupt-/Kontroll-/STOP-Ausgaben berichtet. **H1: {h1['urteil']}; H4: {h4['urteil']}; H1-Gate {result['H1_gate']}.** BEF-0087 rein deskriptiv, **Analyst nicht verblindet**. BEF-0042/0081/0075 nicht schätzbar; Holm-Familie unverändert drei Tests. IPW **nicht schätzbar (Runnerlogik bei Endpunktstopps)** gemäß K-Nachtrag. Keine Modell-/Parameteränderung und kein zweiter Lauf nach Ergebnissicht.

Hauptzahlen und alle Einzelbelege stehen in nachrechnung/B_TESTERGEBNIS_AW142.md, erste Ergebnistabelle; exakte Werte in CSV und Detailanhang. Quelle `{source}`, SHA `{sha(source)}`, results/BEF-ID und sensitivities/i. Der Freeze §0 legt primär <40 %, Sensitivität ≤40 % fest und hat Vorrang vor dem alten darunterstehenden Text. Keine Abweichung des ausgeführten Endpunkts.

## Geänderte/neue Dateien

- nachrechnung/B_TESTERGEBNIS_AW142.md: vollständiger deutscher Bericht mit Fallflüssen, OR/Wald-KI/p/Hierarchie, apparenter ΔAUC/Bootstrap-KI, B4 und sämtlichen Sensitivitäten, F1-F8 je Ergebnis, wörtlicher Messgateakte/Offenlegungen/Schlagmedian-Erratum.
- nachrechnung/b_testergebnis.csv: {len(rows)} Aggregatzeilen mit ungerundeten Zahlen, Quellenpfad/Hash/Schlüssel, vollständigen Fehler-/Missingness-/Fallenmaps.
- werkzeuge/b_test/aw142_vorpruefung.py: nur Hilfsskript auf neue K-Freigabe und autorisiertes Ausgabeziel umgestellt.
- werkzeuge/b_test/aw142_start.ps1 und aw142_ergebnis.py: neue Hilfsstarter/Statusdatei und ausschließlich dokumentarischer Aggregatbericht.
- <local path>: aktualisierte Vorprüfung, synthetische Tests, Status/Logs, bericht_qa.json; alte Dateien nicht gelöscht.
- <local path>: ergebnis.json, zugriffe.jsonl, Ressourcen/Budget, deterministische Bootstrapcheckpoints gemäß ausdrücklichem K-Nachtrag.
- koordination/TESTZUGRIFFE.md: ausschließlich Beginn-/Endezeilen angefügt.
- Diese Abgabe. Auftragsbuch, Befunde, gebundene Dateien und fremde Prozesse unverändert.

## Belege / Prüfung

87 Hashvergleiche vor Start, nach Tests und nach Lauf bestanden. Explizite Testdateiliste aus vorpruefung.json; 103 bestanden, der bekannte einzelne Fehlschlag gemäß K akzeptiert. Testfehlerannahme: fehlende FREIGABE_K.json; tatsächlicher Fehler auf absichtlich fehlender synthetischer Parameterdatei vor produktivem Datenlesen. Keine Teständerung/-überspringung.

Aufrufe: aw142_start.ps1 -Phase preflight; losgelöster Hidden-Start von aw142_start.ps1 -Phase run (unverändertes start.ps1 -Mode runner -Role test -Output <local path>); aw142_start.ps1 -Phase report. Ein Pythonprozess, Auswertungsumgebung, native Threads=1, ≥4-GiB-Gate. Eigene Supervisor-PID 24736, keine Prozesse beendet. Erster Start-Process scheiterte vor Prozessanlage wegen doppeltem Path/PATH; zweiter absoluter Aufruf nach Entfernen von PATH aus der eigenen Shell erfolgreich, keine Testfitfortsetzung. Ressourcen vollständig im Bericht aus ergebnis.json/resources und tests/resources.json.

Bericht SHA `{sha(reportpath)}`; CSV SHA `{sha(csvpath)}`; QA `{sha(OUT/'bericht_qa.json')}`. QA: alle {len(rows)} Zeilen, sechs Hauptteile und sämtliche {len(result['sensitivities'])} Sensitivitätseinträge vorhanden; wortgleiche Messgateakte; gebundene Hashes unverändert. Keine zusätzlichen Fits/Patientenlesungen für Bericht oder QA.

## Offene Fragen / Grenzen

K entscheidet Fachstatus und Manuskriptverwendung. Kandidatenpool-zu-Index-Fluss und Klassenhäufigkeiten gescheiterter Fits teils nicht im Runner exportiert; Basismaskenfluss der Untergruppen ist nicht deren endgültiger Selektionsnenner. Knappes NT-KeyError `names` wird als ungebundener Designparameter offengelegt. Keine zusätzlichen Datenlesungen/Reparaturen. Signalränge 7/8a/8b/9 nicht schätzbar (Skalierung nicht gebunden); EchoNext ausgeschlossen. Der separate nachgelagerte Störtest liegt außerhalb der implementierten B-Runnerteile dieses Auftrags und wurde nicht nachgeholt. Keine Literaturrecherche, keine zusätzlichen Pakete.

## Nebenbefunde / Quellenfehler-Kandidaten

Keine externen Literaturquellen geprüft. Das interne Beschreibungserratum zum normierten, dimensionslosen Schlagmedian ist wörtlich übernommen. Keine weitere Studie/Manuskriptdeutung. Die STOP-Abgabe 18:09Z bleibt im Zugriffsregister nachvollziehbar; Bericht/CSV/Abgabe wurden mit der erfolgreichen Durchführung nach K-Nachtrag ersetzt.

## Selbsteinschätzung der Sicherheit

Hoch für Integrität, einmalige unveränderte Ausführung und getreue Aggregatübertragung. F1-F8 begrenzen die wissenschaftliche Aussage weiterhin; insbesondere informative Missingness, fehlende IPW und ungleiche Messgüte. Statistischer Erfolg ist keine automatische E3-/klinische Freigabe. Auftragserledigung und Abnahme durch K sind getrennt.
'''
    (PROJECT/'koordination/abgaben/AW-142.md').write_text(submission, encoding='utf8')
    print(js(qa))


if __name__ == '__main__':
    main()
