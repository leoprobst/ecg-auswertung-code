"""S5-MC-MED: Ergebnisbericht aus den (bereits maskierten) Ausgaben des Skripts v17. Nur Aggregate, keine Kennungen, keine neuen Rechnungen.
Quellen: <local path>{pilot,bestaetigung}.json und *_status.json, erkenner_protokoll_v16.jsonl, freeze_v16.json, pipeline\\state.json.
Ziel: nachrechnung/S5_MCMED_ERGEBNIS.md. Zahlen werden unveraendert uebernommen (Kleinzellen bleiben "<10")."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
AUSW = Path(str(data_path("s5.auswertung")))
PRIV = Path(str(data_path("s5.analyse_mc_med_s5")))
AUS = PRIV / 'auswertung_v16'
ZIEL = AUSW / 'nachrechnung/S5_MCMED_ERGEBNIS.md'


def f(x, nd=3):
    if x is None:
        return 'n/a'
    if isinstance(x, bool):
        return 'ja' if x else 'nein'
    if isinstance(x, float):
        return f'{x:.{nd}f}'.replace('.', ',')
    if isinstance(x, list):
        return '[' + '; '.join(f(v, nd) for v in x) + ']'
    return str(x)


def ci(d):
    if not isinstance(d, dict):
        return 'n/a'
    return f"{f(d.get('wert'))} ({f(d.get('ci95'))})"


def lade(p):
    p = Path(p)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def tabelle(kopf, zeilen):
    t = ['| ' + ' | '.join(kopf) + ' |', '|' + '|'.join('---' for _ in kopf) + '|']
    t += ['| ' + ' | '.join(f(c) for c in z) + ' |' for z in zeilen]
    return t


def a_abschnitt(g, titel):
    L = [f'### {titel}', '']
    A = (g or {}).get('A') or {}
    z = A.get('zusammenfassung') or {}
    L += tabelle(['Kennzahl', 'Wert'], [
        ['Besuche gesamt', z.get('besuche_gesamt')], ['mit mindestens 3 vollständigen Einheiten (Nenner)', z.get('nenner')],
        ['identifizierbar', z.get('identifizierbar')], ['identifizierbarer Anteil (95-%-Intervall)', ci(z.get('identifizierbar_anteil'))],
        ['Median des Offsets (min)', z.get('offset_median_min')], ['Lageanteil innerhalb ±2 min (Intervall)', ci(z.get('lage_anteil'))],
        ['Populationsminimum bei (min)', z.get('population_minimum_bei_min')], ['mediane Zonenbreite (min)', z.get('breite_median_min')],
        ['Anteil mit Gleichstand', z.get('tie_anteil')]])
    d = (g or {}).get('deutung') or {}
    h0 = d.get('H0') or {}
    L += ['', f"H0 (Tor): erfüllt {f(h0.get('erfuellt'))} (Anteil {f(h0.get('teil_anteil_ident'))}, innerhalb ±2 min {f(h0.get('teil_innerhalb_2'))}, Populationsminimum {f(h0.get('teil_populationsminimum'))}).", '']
    hv = A.get('h0_variabilitaet') or {}
    if hv.get('oben'):
        L += ['H0 nach HF-Variabilität (Terzile der HF-Standardabweichung der Einheiten; entscheidet das Tor nicht):', '']
        L += tabelle(['Terzil', 'Besuche', 'identifizierbar', 'Anteil', 'innerhalb ±2 min (Anteil)'], [[k, hv[k].get('besuche'), hv[k].get('identifizierbar'), hv[k].get('anteil'), hv[k].get('innerhalb_2_anteil')] for k in ('unten', 'mitte', 'oben') if k in hv])
        L += ['', f"Oberes Terzil erfüllt H0 (alle drei Teile): {f(hv.get('oben_erfuellt_h0'))}.", '']
    nk3 = A.get('nk3') or {}
    if nk3:
        L += ['Dichte-Kurve (NK3, Monitorreferenz ausgedünnt; Nenner sinkt mit dem Abstand):', '']
        L += tabelle(['Referenzabstand (min)', 'Nenner (Mittel)', 'identifizierbarer Anteil (95-%-Intervall)'], [[k, nk3[k].get('nenner_mittel'), ci(nk3[k].get('anteil'))] for k in sorted(nk3, key=lambda x: float(x))])
        L += ['']
    nk1, nk2, nk4 = A.get('nk1') or {}, A.get('nk2') or {}, A.get('nk4') or {}
    L += tabelle(['Kontrolle', 'Ergebnis'], [
        ['NK1 (Permutation im Besuch), identifizierbar höchstens', nk1.get('identifizierbar_max')], ['NK1 Anteil höchstens', nk1.get('identifizierbar_anteil_max')],
        ['NK2 (anderer Patient), Falsch-Identifizierung Mittel', nk2.get('falsch_identifiziert_anteil_mittel')], ['NK2 Nenner Mittel / ohne Partner Mittel', f"{f(nk2.get('nenner_mittel'))} / {f(nk2.get('ohne_partner_mittel'))}"],
        ['NK4 (HF × 1,15), identifizierbarer Anteil', ci((nk4.get('identifizierbar_anteil')))]])
    pk2 = A.get('pk2_verschiebung') or {}
    if pk2:
        L += ['', 'Wiederfindung eingefügter Verschiebungen (PK2, differenziell):', '']
        L += tabelle(['Verschiebung (min)', 'beide identifizierbar', 'Median |Fehler| (min)', 'absolut Median |Fehler| (min)', 'S5-Regel erfüllt'],
                     [[k, v.get('beide_identifizierbar'), v.get('median_abs_fehler_min'), v.get('absolut_median_abs_fehler_min'), v.get('erfolg_s5_regel')] for k, v in sorted(pk2.items(), key=lambda kv: float(kv[0]))])
    L += ['', 'Hypothesen-Etiketten (vorab festgelegte Schwellen): ' + ', '.join(f'{k} = {f(d[k])}' for k in ('H3', 'H4', 'H5', 'H6', 'NK4') if k in d) + '.', '']
    sa = ((g or {}).get('sensitivitaeten') or {}).get('A') or {}
    if sa:
        L += ['Sensitivitäten Beziehung A:', '']
        L += tabelle(['Variante', 'Nenner', 'identifizierbar', 'Anteil (Intervall)'], [[k, v.get('nenner'), v.get('identifizierbar'), ci(v.get('anteil'))] for k, v in sa.items()])
        L += ['']
    return L


def b_abschnitt(g, titel):
    L = [f'### {titel}', '']
    B = (g or {}).get('B') or {}
    if B.get('vor_einfrieren') or not B:
        return L + ['Keine B-Schätzung (vor dem Einfrieren oder nicht vorhanden).', '']
    if B.get('gesperrt'):
        L += [f"B gesperrt ({B.get('gesperrt')}): keine Beschreibung."]
    beschr = B.get('beschreibung_ohne_urteil') or (B if 'zusammenfassung' in B else None)
    if beschr:
        z = beschr.get('zusammenfassung') or {}
        ref = z.get('referenz') or {}
        L += ['Beschreibung der Beziehung B' + (' (Nenner unter Z2: ohne ZB-Urteil, ohne Kontrollen)' if B.get('nicht_auswertbar') else '') + ':', '']
        L += tabelle(['Kennzahl', 'Wert'], [
            ['Besuche gesamt', z.get('besuche_gesamt')], ['mit mindestens 3 vollständigen Einheiten (roh)', z.get('mit_mind_3_roh')], ['durch Ankerbedingung ausgeschlossen', z.get('ausgeschlossen_durch_anker')],
            ['Nenner', z.get('nenner')], ['identifizierbar', z.get('identifizierbar')], ['identifizierbarer Anteil (95-%-Intervall)', ci(z.get('identifizierbar_anteil'))],
            ['Median des Offsets (min)', z.get('offset_median_min')], ['Lageanteil innerhalb ±5 min (Intervall)', ci(z.get('lage_anteil'))], ['Anteil mit Gleichstand', z.get('tie_anteil')],
            ['Anker bei D = 0, Median (Quartile)', f"{f(ref.get('anker_d0_median'))} ({f(ref.get('anker_d0_quartile'))})"], ['zeitliche Spannweite der Anker, Median (min)', ref.get('spannweite_min_median')],
            ['Referenzwechsel, Median', ref.get('wechsel_median')], ['Anteil informationsloser Einheiten (Mittel)', ref.get('einheiten_ohne_wechsel_anteil_mittel')]])
        L += ['']
    for k, name in (('pkb', 'PK-B'), ('nks', 'NK-S')):
        if B.get(k):
            L += [f'{name} liegt vor (siehe Ausgabedatei); Zusammenfassung:', '']
            if k == 'pkb':
                e0 = B[k].get('e0') or {}
                L += tabelle(['Größe', 'Wert'], [['E0 erfüllt', e0.get('erfuellt')], ['E0 beurteilbar', e0.get('beurteilbar')], ['Lage auflösbar (E0l)', e0.get('lage_aufloesbar')], ['Abdeckung der Chartpunkte durch Monitor', (B[k].get('abdeckung') or {}).get('anteil')],
                                                 ['Überlapp B-Nenner und PK-B-Nenner', (B[k].get('kontrollgueltigkeit') or {}).get('anteil_des_b_nenners_im_pkb_nenner')]])
            else:
                L += tabelle(['Größe', 'Wert'], [['rotierbare Besuche', B[k].get('besuche_rotierbar')], ['Patienten', B[k].get('patienten_rotierbar')], ['Anteil identifizierbar Daten', ci((B[k].get('identifizierbar_anteil') or {}).get('daten'))],
                                                 ['Anteil identifizierbar Null', ci((B[k].get('identifizierbar_anteil') or {}).get('kontrolle'))], ['Differenz', ci((B[k].get('identifizierbar_anteil') or {}).get('differenz'))]])
            L += ['']
    zb = ((g or {}).get('deutung') or {}).get('ZB') or {}
    L += [f"**ZB-Urteil:** {zb.get('urteil', 'kein Eintrag')}", '']
    sb = ((g or {}).get('sensitivitaeten') or {}).get('B') or {}
    if sb:
        L += ['Sensitivitäten Beziehung B:', '']
        L += tabelle(['Variante', 'Nenner', 'identifizierbar', 'Anteil (Intervall)'], [[k, v.get('nenner'), v.get('identifizierbar'), ci(v.get('anteil'))] for k, v in sb.items()])
        L += ['']
    return L


def status_abschnitt(g):
    s = (g or {}).get('status') or {}
    nd = s.get('nenner_detail') or {}
    L = tabelle(['Größe', 'Wert'], [
        ['Zweig / Status', f"{s.get('zweig')} / {s.get('status')}"], ['Urteilsform', s.get('urteilsform')],
        ['Besuche mit Einheiten im längsten Segment (Population der Auswertung)', nd.get('besuche_gesamt')], ['mit mindestens 1 vollständigen Einheit', nd.get('mit_mind_1_vollstaendig')],
        ['mit mindestens 3 vollständigen Einheiten (roh)', nd.get('mit_mind_3_roh')], ['B-Nenner (nach Ankerbedingung ≥ 2)', nd.get('nenner')], ['durch Ankerbedingung ausgeschlossen', nd.get('ausgeschlossen_durch_anker')],
        ['rotierbarer Vergleichsnenner', nd.get('rotierbarer_nenner')], ['Patienten im rotierbaren Nenner', nd.get('rotierbare_patienten')], ['nicht rotierbar (Gründe)', json.dumps(nd.get('nicht_rotierbar_gruende'), ensure_ascii=False)]])
    return L + ['']


def schreibe_bericht():
    pilot, best = lade(AUS / 'pilot.json'), lade(AUS / 'bestaetigung.json')
    fz = lade(PRIV / 'freeze_v16.json') or {}
    lock = lade(PRIV / 'spec_lock_v16.json') or {}
    prot = [json.loads(z) for z in (PRIV / 'erkenner_protokoll_v16.jsonl').read_text(encoding='utf-8').splitlines() if z.strip()] if (PRIV / 'erkenner_protokoll_v16.jsonl').exists() else []
    stt = lade(PRIV / 'pipeline/state.json') or {}
    L = ['# S5-MC-MED: Ergebnisbericht (externe Prüfung lokaler Zuordnung an MC-MED)', '',
         f'Erzeugt {dt.datetime.now().strftime("%d.%m.%Y %H:%M")} automatisch aus den maskierten Ausgaben der gebundenen Auswertung (Spezifikation v1.7, Skript v17). **Nur Aggregate**; Zählungen von 1 bis 9 erscheinen als „<10“; keine Kennungen; Testsplits wurden nicht geöffnet. '
         'Das Ergebnis ist eine **explorative externe Prüfung lokaler Zuordnung innerhalb ±15 Minuten**, keine Prüfung der ursprünglichen S5-Aussage innerhalb ±60 Minuten und kein Beleg für Gerätuhren.', '',
         '## 1. Bindung und Ablauf', '',
         f"Gebunden am {lock.get('gesperrt_utc', 'n/a')} (UTC); Skript `code_sha256` {lock.get('code_sha256', 'n/a')}; Spezifikation `spec_sha256` {lock.get('spec_sha256', 'n/a')}. Eingefroren: {fz.get('eingefroren_utc', 'n/a')} (Erkennerversion {fz.get('erkenner_version', 'n/a')}). "
         f"Neustarts des Treibers durch den Wachprozess: {stt.get('neustarts', 0)}. Schrittdauern siehe `koordination/abgaben/S5_MCMED_ZWISCHENSTAND.md`.", '',
         '## 2. Erkennerwahl (technische Güte auf der vollständigen Val-Grundgesamtheit)', '']
    zeilen = []
    for e in prot:
        k = e.get('kriterium') or {}
        zeilen.append([e.get('ev', '')[:8], (e.get('extraktionsparameter') or {}).get('rr_cv_max'), (e.get('extraktionsparameter') or {}).get('abdeckung_A_min_s'), k.get('anteil_innerhalb'), k.get('ausbeute'),
                       k.get('fenster_gesamt'), e.get('lesefehler_fenster') if (e.get('lesefehler_fenster') or 0) >= 10 or not e.get('lesefehler_fenster') else '<10', k.get('kriterium_erfuellt'), 'ja' if e.get('ev') == fz.get('erkenner_version') else ''])
    L += tabelle(['Erkennerversion', 'rr_cv_max', 'Abdeckung min (s)', 'Übereinstimmung innerhalb 5/min', 'Ausbeute', 'Fenster', 'Lesefehler-Fenster', 'Kriterium erfüllt', 'gewählt'], zeilen) + ['']
    L += ['## 3. Pilot (Val)', '', 'Der Pilot dient der Fehlersuche und gibt kein Urteil.', ''] + status_abschnitt(pilot) + a_abschnitt(pilot, 'Beziehung A (technische Kontrolle)') + b_abschnitt(pilot, 'Beziehung B')
    L += ['## 4. Bestätigung (Train)', ''] + status_abschnitt(best) + a_abschnitt(best, 'Beziehung A (technische Kontrolle, ohne Beweiskraft für S5)') + b_abschnitt(best, 'Beziehung B (Primärfrage)')
    L += ['## 5. Lesehinweise und Grenzen', '',
          '- Beziehung A (Monitor-HF gegen Wellenform-HF) hat konstruktionsbedingt den Sollversatz 0 und ist eine technische Kontrolle; sie sagt nichts über die Uhren der MIMIC-Geräte.',
          '- Beziehung B vermischt Uhrdifferenz, Dokumentationsverzug und Messverfahren der Chart-HF; ein positiver Offset ist mit Dokumentationsverzug vereinbar. Die Aussage bleibt auf lokale Zuordnung im Fenster ±15 min begrenzt.',
          '- Je Besuch wurde nur das längste Einzelsegment verwendet (Spezifikation v1.6); der Nenner gilt für diese Teilpopulation.',
          '- Ist der rotierbare Vergleichsnenner kleiner als Zweig Z2 (1.000 Besuche und 500 Patienten), gibt es kein ZB-Urteil; B wird dann nur beschrieben.',
          '- Fehlende Eintragungen („n/a“) bedeuten, dass die jeweilige Größe in diesem Zweig nicht berechnet wurde.', '']
    ZIEL.write_text('\n'.join(L) + '\n', encoding='utf-8')
    return ZIEL


if __name__ == '__main__':
    print(schreibe_bericht())
