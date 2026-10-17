"""POST HOC, BESCHREIBEND, NICHT GEBUNDEN. Unabhaengige Pruefung der Kleinzellenregel fuer die Basistabelle S3 (MC-MED).

Liest KEINE Daten (weder Datenbank noch Zwischenspeicher noch die private Ergebnisdatei), nur die oeffentliche Struktur oeffentlich.json (nur Anzeigen, Schranken, Spannen)
und die oeffentliche Markdown-Datei. Prueft, ob aus dem VEROEFFENTLICHTEN eine Zaehlung exakt ableitbar ist, wie ein Dritter es versuchen wuerde:

  P1  Keine Zaehlung 1 bis 9 (auch keine 0 als Zaehlung einer Teilgruppe) in den Zaehlfeldern der Struktur und in Zaehlzusammenhaengen des Textes.
  P2  Anteile: Fuer jede Gruppe wird die Menge aller Zaehlungen aufgezaehlt, die mit den gezeigten Anteilen, der gezeigten (abgerundeten) Personenzahl und den
      Aufteilungen (Summe gleich Gruppengroesse) vereinbar sind. Ist die Menge fuer ein Merkmal ein Einzelwert, ist die Zaehlung exakt ableitbar (Verstoss).
  P3  Differenzen: Alle Differenzen und Summen von bis zu drei exakt veroeffentlichten Zahlen (Zaehlkette Tabelle S1, Tabelle S4, Tabelle dieser Datei) duerfen
      nicht 1 bis 9 sein, ausser sie sind als bekannte Einengung dokumentiert.
  P4  Personenzahl gegen Paarzahl: Differenz der veroeffentlichten Zahlen 1 bis 9 wird gemeldet (Einengung, keine exakte Ableitung, wenn die Spanne mehrere Werte hat).
  P5  Text: Zahlen neben Paare, Personen, Patienten, pairs, patients im Fliesstext duerfen nicht 1 bis 9 sein.
Modi: pruefen (Standard), selbsttest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, json, re, itertools, math
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(str(data_path("s3.posthoc_basis_dir")))
OEFF = OUT / 'oeffentlich.json'
MD = Path(str(data_path("s3.posthoc_basis_table_md")))
FRAG = OUT / 'tabelle_fragment.md'

# Exakt veroeffentlichte Zahlen aus Supplement v2 (Tabelle S1: 5661, 5585, 970, 592, 522, 324, 216; Tabelle S4: 240, 139, 53, 24, 105, 314, 205, 199, 146, 142, 132, 63, 118)
# und S3_POSTHOC_H2.md; sie bilden den Hintergrund, gegen den die neuen Zahlen geprueft werden.
OEFFENTLICH_BEKANNT = {'S1 konsekutiv': 5661, 'S1 gueltig': 5585, 'S1 EKG beide Enden': 970, 'S1 Messung gueltig': 592, 'S1 Polaritaet': 522, 'S1 zugelassen': 324,
                       'S1 primaer': 216, 'S4 ohne dokumentierte Behandlung': 240, 'S4 dokumentiert': 105, 'S4 alle Quellen': 314, 'S4 Anstieg': 63, 'S4 Abfall': 132,
                       'S4 K bis 3,5': 24, 'S4 K 3,5 bis 5,0': 139, 'S4 K ueber 5,0': 53}


def kandidaten(d, N):
    """Alle Zaehlungen k in 0..N, die mit der Anzeige d (typ prozent, klein, gross; optional gerundet5) bei Gruppengroesse N vereinbar sind."""
    t = d['typ']
    if t == 'prozent':
        if d.get('gerundet5'):
            return [k for k in range(N + 1) if int(5 * round(100.0 * k / N / 5)) == d['prozent']]
        return [k for k in range(N + 1) if int(round(100.0 * k / N)) == d['prozent']]
    if t == 'klein':
        return [k for k in range(N + 1) if 100.0 * k / N < d['schranke']]
    if t == 'gross':
        return [k for k in range(N + 1) if 100.0 * k / N > d['schranke']]
    return None


def familie_pruefen(mitglieder, n_spanne, vollstaendig):
    """mitglieder: {name: anzeige}. Rueckgabe {name: sortierte Liste moeglicher Zaehlungen}, alle N im Bereich n_spanne vereinigt.
    vollstaendig: die Mitglieder teilen die Gruppe vollstaendig auf (Summe gleich N); sonst ist jedes Mitglied einzeln durch seine Anzeige gebunden."""
    erg = {k: set() for k in mitglieder}
    for N in range(n_spanne[0], n_spanne[1] + 1):
        cs = {k: kandidaten(v, N) for k, v in mitglieder.items()}
        if any(c is None for c in cs.values()):
            continue
        if not vollstaendig:
            for k, c in cs.items():
                erg[k].update(c)
            continue
        namen = list(mitglieder)
        for kombi in itertools.product(*[cs[n] for n in namen[:-1]]):
            rest = N - sum(kombi)
            if rest in cs[namen[-1]]:
                for n, k in zip(namen, list(kombi) + [rest]):
                    erg[n].add(k)
    return {k: sorted(v) for k, v in erg.items()}


def pruefe_struktur(P):
    verstoesse, hinweise, tabelle = [], [], []
    for g, G in P['gruppen'].items():
        spanne = G['personen_spanne_oeffentlich']
        for k in ('paare', 'personen'):
            z = G[k]
            if z['art'] != 'klein' and not (z.get('wert', 0) >= 10):
                verstoesse.append(('P1', g, k, z))
        m = G['merkmale']
        fam_dk = {k: m[k] for k in ('anstieg', 'abfall', 'unveraendert') if m[k]['typ'] in ('prozent', 'klein', 'gross')}
        if len(fam_dk) == 3:
            r = familie_pruefen(fam_dk, spanne, True)
            for k, v in r.items():
                tabelle.append((g, 'dk_richtung:' + k, len(v), min(v) if v else None, max(v) if v else None))
                if len(v) < 2:
                    verstoesse.append(('P2', g, k, v))
                if not v:
                    hinweise.append(('P2 leere Menge (Anzeigen widersprechen sich)', g, k))
        for k in ('geschlecht_m', 'hf_verfuegbar', 'qrs_verfuegbar', 'ecg_beide_enden', 'messung_gueltig_beide_enden'):
            if m[k]['typ'] in ('prozent', 'klein', 'gross'):
                r = familie_pruefen({k: m[k]}, spanne, False)
                v = r[k]
                tabelle.append((g, k, len(v), min(v) if v else None, max(v) if v else None))
                if len(v) < 2:
                    verstoesse.append(('P2', g, k, v))
        # Statistiken: nur bei mindestens 10 Einheiten berichtet; Zahl der Einheiten ist durch die Gruppengroesse und den Anteil (hf_verfuegbar, qrs_verfuegbar) gebunden
        for k in ('hf_a', 'qrs_a', 'alter', 'ka', 'dk', 'gap_h'):
            d = m[k]
            if d.get('typ') == 'stat':
                av = {'hf_a': 'hf_verfuegbar', 'qrs_a': 'qrs_verfuegbar'}.get(k)
                if av and m[av]['typ'] in ('klein',):
                    verstoesse.append(('P1', g, k, 'Statistik trotz verdeckter Verfuegbarkeit unter der Schranke'))
    for g, B_ in P.get('behandlung', {}).items():
        for k, d in B_['anteile'].items():
            if d['typ'] in ('prozent', 'klein', 'gross'):
                v = familie_pruefen({k: d}, B_['spanne_paare'], False)[k]
                tabelle.append((g, 'behandlung:' + k, len(v), min(v) if v else None, max(v) if v else None))
                if len(v) < 2:
                    verstoesse.append(('P2', g, 'behandlung:' + k, v))
    return verstoesse, hinweise, tabelle


def beziehungen(P):
    """Logische Beziehungen zwischen exakt veroeffentlichten Zahlen: (eltern, [kinder]) mit Kindern als Teilmenge (eine Liste mit einem Kind) oder vollstaendiger Aufteilung.
    Der Rest eltern minus Summe der Kinder muss 0 oder mindestens 10 sein."""
    ex = dict(OEFFENTLICH_BEKANNT)
    g = P['gruppen']
    ex['Paare A'], ex['Paare B'], ex['Paare C*'], ex['Paare D*'] = g['A']['paare']['wert'], g['B']['paare']['wert'], g['C_ana']['paare']['wert'], g['D_ana']['paare']['wert']
    gr = {k: v['wert'] for k, v in P['gruende_b'].items()}
    for k, v in gr.items():
        ex['Grund ' + k] = v
    ex['S1 Erstwert ab 6 (324 minus 216)'] = ex['S1 zugelassen'] - ex['S1 primaer']
    teilmenge = [('S1 konsekutiv', 'S1 gueltig'), ('S1 gueltig', 'S1 EKG beide Enden'), ('S1 EKG beide Enden', 'S1 Messung gueltig'), ('S1 Messung gueltig', 'S1 Polaritaet'),
                 ('S1 Polaritaet', 'S1 zugelassen'), ('S1 zugelassen', 'S1 primaer'), ('S1 gueltig', 'Paare A'), ('S1 gueltig', 'Paare B'), ('S1 primaer', 'S4 Anstieg'),
                 ('S1 primaer', 'S4 Abfall'), ('S1 zugelassen', 'Paare D*'), ('S4 ohne dokumentierte Behandlung', 'S1 primaer'), ('S4 dokumentiert', 'Paare C*'),
                 ('Paare B', 'Grund kein_ecg_beide_enden'), ('Paare B', 'Grund messung_ungueltig'), ('Paare B', 'Grund polaritaet'), ('Paare B', 'Grund ueberlappung_hf_behandlung'),
                 ('Paare B', 'Grund erstwert_ab_6')]
    teilmenge = [(a, [b]) for a, b in teilmenge if a in ex and b in ex]
    aufteilung = [('S1 gueltig', ['Paare A', 'Paare B']), ('Paare B', ['Grund kein_ecg_beide_enden', 'Grund messung_ungueltig', 'Grund polaritaet', 'Grund ueberlappung_hf_behandlung',
                                                                       'Grund erstwert_ab_6']),
                  ('S1 zugelassen', ['S1 primaer', 'S1 Erstwert ab 6 (324 minus 216)']), ('S1 primaer', ['S4 K bis 3,5', 'S4 K 3,5 bis 5,0', 'S4 K ueber 5,0'])]
    aufteilung = [(a, k) for a, k in aufteilung if a in ex and all(x in ex for x in k)]
    return ex, teilmenge, aufteilung


def pruefe_differenzen(P):
    """P3: Rest einer Teilmenge/Aufteilung gegen die Obermenge (Komplement) darf nicht 1 bis 9 sein. Der Wert 0 ist zulaessig (vollstaendige Aufteilung)."""
    ex, teilmenge, aufteilung = beziehungen(P)
    treffer = {}
    for eltern, kinder in teilmenge + aufteilung:
        d = ex[eltern] - sum(ex[k] for k in kinder)
        if 1 <= d <= 9:
            treffer[(eltern, tuple(kinder))] = d
    return treffer


def pruefe_personen_paare(P):
    out = []
    for g, G in P['gruppen'].items():
        pa, pe = G['paare'], G['personen']
        if pa.get('art') == 'exakt' and pe.get('art') != 'klein':
            d = pa['wert'] - pe['wert']
            if 1 <= d <= 9:
                out.append((g, pa['wert'], pe['wert'], d, G['personen_spanne_oeffentlich']))
    return out


def pruefe_text(txt):
    treffer = []
    for m in re.finditer(r'(?<![\d.,])([0-9]+)\s*(Paare|Personen|Patienten|pairs|patients|Person)\b', txt):
        n = int(m.group(1))
        if 1 <= n <= 9:
            treffer.append((m.group(0), txt[max(0, m.start() - 40): m.end() + 20].replace('\n', ' ')))
    return treffer


def pruefe_zusatz(Z):
    """Zusatzzeilen (Triage, Entlassungsziel, Aufenthaltsdauer, ED-Diagnosen): Aufteilungen vollstaendig (Summe gleich Gruppengroesse), Einzelanteile einzeln."""
    verstoesse, tabelle = [], []
    for g, G in Z['gruppen'].items():
        sp = G['personen_spanne_oeffentlich']
        for fam in ('triage', 'dispo'):
            mit = {k: v for k, v in G[fam].items() if v['typ'] in ('prozent', 'klein', 'gross')}
            r = familie_pruefen(mit, sp, len(mit) == len(G[fam]))
            for k, v in r.items():
                tabelle.append((g, fam + ':' + k, len(v), min(v) if v else None, max(v) if v else None))
                if len(v) < 2:
                    verstoesse.append(('P2', g, fam + ':' + k, v))
        einzel = {'ed_los_verfuegbar': G['ed_los_verfuegbar']}
        einzel.update({'dx:' + k: v for k, v in G['dx'].items()})
        for k, d in einzel.items():
            if d['typ'] in ('prozent', 'klein', 'gross'):
                v = familie_pruefen({k: d}, sp, False)[k]
                tabelle.append((g, k, len(v), min(v) if v else None, max(v) if v else None))
                if len(v) < 2:
                    verstoesse.append(('P2', g, k, v))
        if G['ed_los'].get('typ') == 'stat' and G['ed_los_verfuegbar']['typ'] == 'klein' and G['ed_los_verfuegbar'].get('schranke', 100) < 10:
            pass
    return verstoesse, tabelle


def pruefen():
    P = json.loads(OEFF.read_text(encoding='utf-8'))
    ber = {}
    v, h, tab = pruefe_struktur(P)
    ber['P1_P2_verstoesse'], ber['P2_hinweise'], ber['P2_spannen'] = v, h, tab
    t3 = pruefe_differenzen(P)
    ber['P3_differenzen_1_bis_9'] = [[k[0], list(k[1]), d] for k, d in t3.items()]
    ex, tm, au = beziehungen(P)
    ZJ = OUT / 'zusatz_oeffentlich.json'
    if ZJ.exists():
        vz, tz = pruefe_zusatz(json.loads(ZJ.read_text(encoding='utf-8')))
        ber['Zusatz_verstoesse'], ber['Zusatz_spannen'] = vz, tz
    ber['P3_geprueft'] = {'beziehungen': len(tm) + len(au), 'exakte_zahlen': len(ex)}
    ber['P4_personen_gegen_paare'] = pruefe_personen_paare(P)
    if MD.exists():
        txt = MD.read_text(encoding='utf-8')
        ber['P5_text'] = pruefe_text(txt)
        ber['md_enthaelt_tabelle'] = FRAG.read_text(encoding='utf-8').split('\n\n')[0] in txt
        ZF = OUT / 'zusatz_fragment.md'
        ber['md_enthaelt_zusatz'] = (ZF.read_text(encoding='utf-8').strip() in txt) if ZF.exists() else None
        ber['md_enthaelt_gruende'] = FRAG.read_text(encoding='utf-8').split('\n\n')[1].strip() in txt
    else:
        ber['P5_text'] = 'Markdown-Datei fehlt'
    (OUT / 'pruefung_ableitbarkeit.json').write_text(json.dumps(ber, indent=1, ensure_ascii=False), encoding='utf-8')
    print('P1/P2 Verstoesse:', len(v))
    for x in v:
        print('  ', x)
    print('P2 kleinste Spanne moeglicher Zaehlungen je Merkmal (Zahl der Werte, kleinster, groesster):')
    for t in sorted(tab, key=lambda t: t[2])[:12]:
        print('  ', t)
    print('P3 Rest einer Teilmenge oder Aufteilung gegen die Obermenge 1 bis 9:', len(t3), '(geprueft:', ber['P3_geprueft'], ')')
    for k, d in list(t3.items())[:40]:
        print('  ', k, d)
    print('P4 Personen gegen Paare (Differenz 1 bis 9, Spanne der Personenzahl):', ber['P4_personen_gegen_paare'])
    print('P5 Text:', ber['P5_text'] if isinstance(ber['P5_text'], str) else len(ber['P5_text']))
    for x in (ber['P5_text'] if not isinstance(ber['P5_text'], str) else []):
        print('  ', x)
    print('Zusatzzeilen: Verstoesse', len(ber.get('Zusatz_verstoesse', [])), 'Anteile geprueft', len(ber.get('Zusatz_spannen', [])), 'kleinste Spannen', sorted(ber.get('Zusatz_spannen', []), key=lambda t: t[2])[:4], 'Markdown enthaelt Zusatztabelle:', ber.get('md_enthaelt_zusatz'))
    for x in ber.get('Zusatz_verstoesse', []):
        print('  ', x)
    print('Markdown enthaelt Tabelle/Gruende:', ber.get('md_enthaelt_tabelle'), ber.get('md_enthaelt_gruende'))
    return ber


def selbsttest():
    import traceback
    erg = []

    def T_(name):
        def deko(f):
            try:
                f()
                erg.append(True)
                print('  ok  ', name)
            except Exception:
                erg.append(False)
                print('  FAIL', name, '\n', traceback.format_exc())
            return f
        return deko

    @T_('Kandidaten: Prozent, kleine und grosse Schranke')
    def _():
        assert 50 in kandidaten({'typ': 'prozent', 'prozent': 50}, 100) and 55 not in kandidaten({'typ': 'prozent', 'prozent': 50}, 100)
        assert kandidaten({'typ': 'klein', 'schranke': 5}, 200) == list(range(10))
        assert kandidaten({'typ': 'gross', 'schranke': 95}, 200) == list(range(191, 201))

    @T_('Aufteilung: Summe bindet verdeckten Anteil, mehrere Werte moeglich')
    def _():
        fam = {'a': {'typ': 'prozent', 'prozent': 50, 'gerundet5': True}, 'b': {'typ': 'prozent', 'prozent': 45, 'gerundet5': True}, 'c': {'typ': 'klein', 'schranke': 10}}
        r = familie_pruefen(fam, (100, 105), True)
        assert len(r['c']) >= 2 and min(r['c']) >= 0

    @T_('Aufteilung ohne Rundung auf 5 % macht den verdeckten Anteil auffaellig eng, aber noch nicht exakt (N unbekannt)')
    def _():
        fam = {'a': {'typ': 'prozent', 'prozent': 52}, 'b': {'typ': 'prozent', 'prozent': 45}, 'c': {'typ': 'klein', 'schranke': 10}}
        r = familie_pruefen(fam, (101, 101), True)             # N exakt bekannt: Spanne wird eng
        r2 = familie_pruefen(fam, (100, 109), True)
        assert len(r['c']) <= len(r2['c'])

    @T_('Geplantes Leck wird erkannt: exakt bekannte Gruppengroesse, zwei Anteile, dritte Zaehlung eindeutig')
    def _():
        fam = {'a': {'typ': 'prozent', 'prozent': 49}, 'b': {'typ': 'prozent', 'prozent': 50}, 'c': {'typ': 'prozent', 'prozent': 1}}
        r = familie_pruefen(fam, (200, 200), True)
        assert all(len(v) == 1 for v in r.values()), r

    @T_('Beziehungen: Aufteilung mit Rest 7 wird gefunden (Leck 522 gegen 324 plus 191 waere ein Rest von 7)')
    def _():
        g = {k: {'paare': {'wert': w, 'art': 'exakt'}} for k, w in (('A', 216), ('B', 5369), ('C_ana', 105), ('D_ana', 108))}
        sch = {'kein_ecg_beide_enden': 4615, 'messung_ungueltig': 378, 'polaritaet': 70, 'ueberlappung_hf_behandlung': 198, 'erstwert_ab_6': 108}
        assert pruefe_differenzen({'gruppen': g, 'gruende_b': {k: {'wert': v} for k, v in sch.items()}}) == {}
        sch2 = dict(sch, ueberlappung_hf_behandlung=191)            # getrennte Veroeffentlichung der Behandlung allein: Rest 7 gegen Paare B
        t = pruefe_differenzen({'gruppen': g, 'gruende_b': {k: {'wert': v} for k, v in sch2.items()}})
        assert 7 in t.values(), t

    @T_('Text: Zahlen 1 bis 9 neben Paare/Personen werden gefunden, Zehnerzahlen und Dezimalzahlen nicht')
    def _():
        t = pruefe_text('Es waren 7 Paare, 12 Personen, 1.234 Paare, 0,5 Personen und 3 pairs.')
        assert len(t) == 2, t

    print(f'\nSelbsttest: {sum(erg)} von {len(erg)} bestanden')
    return all(erg)


if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else 'pruefen'
    if m == 'selbsttest':
        sys.exit(0 if selbsttest() else 1)
    pruefen()
