"""POST HOC, NACH SICHT DER ERGEBNISSE, EXPLORATIV, NICHT GEBUNDEN.

Baut aus der privaten Ausgabe <local path> die oeffentlich taugliche Fassung (nur Aggregate) und Markdown-Bausteine.
Liest KEINE Daten (weder die Datenbank noch den Zwischenspeicher), nur die private Ergebnisdatei. Kleinzellenregel: Zaehlungen 1 bis 9 als <10; jede Zahl, aus deren
Differenz zu einer anderen veroeffentlichten Zahl (Teilmenge, Partition, Komplement) eine Zahl 1 bis 9 folgen wuerde, wird auf Zehner abgerundet und markiert. Personenzahlen
werden nicht veroeffentlicht (Paare minus Personen waere eine Zaehlung unter 10). Zaehlungen von Monte-Carlo-Wiederholungen (Seeds) und von Analysen sind keine Personenzaehlungen.
Modi: bauen (Standard), selbsttest.
"""
import sys, json, math, re
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_posthoc_h2 as P

OUT = P.OUT
EIN = OUT / 'posthoc_h2.json'
OEFF = OUT / 'oeffentlich.json'
FRAG = OUT / 'fragmente.md'
GEBUNDEN = P.JSON_GEBUNDEN


def z(x, d=3):
    """Zahl mit deutschem Komma."""
    if x is None:
        return 'n. b.'
    return f'{x:.{d}f}'.replace('.', ',').replace('-', '\u2212') if False else f'{x:.{d}f}'.replace('.', ',')


def iv(lo, hi, d=3):
    return f'{z(lo, d)} bis {z(hi, d)}'


def kat_text(k):
    return {'positiv_ueber_grenze': 'positiv über Grenze', 'positiv_grenze_offen': 'positiv, Grenze offen', 'unpraezise': 'unpräzise', 'grenze_ausgeschlossen': 'Grenze ausgeschlossen',
            'positiv_unter_grenze': 'positiv unter Grenze', 'negativ': 'negativ', 'nicht_beurteilbar': 'nicht beurteilbar'}.get(k, k)


# ----------------------------------------------------------------------------------------------------------------------------- Zaehlungen
BEZIEHUNGEN = [('anstieg', 'primaer'), ('abfall', 'primaer'), ('ohne_einfl', 'primaer'), ('mit_bik', 'primaer'), ('abs_dhf10', 'primaer'), ('eine_je_person', 'primaer'),
               ('na', 'primaer'), ('nk1', 'primaer'), ('train', 'primaer'), ('val_test', 'primaer'), ('d1', 'primaer'), ('primaer', 'ohne_dok'), ('primaer', 'alle_erst'),
               ('ohne_dok', 'alle_erst'), ('primaer', 'alle_quellen'), ('s_hoch', 'primaer'), ('s_unten', 'primaer'), ('stufe_ab_minus1', 'abfall'), ('stufe_ab_0_5', 'primaer'),
               ('val', 'val_test'), ('test', 'val_test')]
PARTITIONEN = [('primaer', ['anstieg', 'abfall']), ('primaer', ['train', 'val', 'test']), ('primaer', ['bereich_unten', 'bereich_mitte', 'bereich_oben']),
               ('primaer', ['stufe_ab_minus1', 'stufe_minus1_minus05', 'stufe_klein', 'stufe_ab_0_5']), ('val_test', ['val', 'test'])]


def zaehlungen_bauen(R):
    S = R['sensitivitaeten']
    K = R['klinisch_h2']['individuell_je_kaliumstufe']
    sp = R['stabilitaet_je_split']
    br = S['ausgangsbereich']
    Z = {'primaer': R['rahmen']['zeilen_h2'], 'anstieg': S['nur_anstieg']['n'], 'abfall': S['nur_abfall']['n'], 'ohne_einfl': S['einfluss']['ohne_einflussreiche']['n'],
         'n_einflussreich': S['einfluss']['n_einflussreich'], 'mit_bik': S['mit_bikarbonat']['n'], 'abs_dhf10': S['nur_abs_dhf_unter_10']['n'],
         'eine_je_person': S['eine_beobachtung_je_person']['n'], 'na': R['natrium_steilheit']['n'], 'nk1': R['nk1_steilheit']['n_paare'], 'nk2': R['nk2_steilheit']['n'],
         'train': sp['train']['n'], 'val': sp['val']['n'], 'test': sp['test']['n'], 'val_test': sp['val_und_test_zusammen']['n'], 'd1': S['d1_gleiche_paare']['gemeinsame_paare'],
         'ohne_dok': S['ohne_dokumentierte_behandlung']['primaerpaare'], 'alle_erst': S['alle_erstwerte']['primaerpaare'], 'alle_quellen': S['alle_quellen']['primaerpaare'],
         'dokumentiert': S['behandlung_dokumentiert']['paare'], 'bereich_unten': br['k1_bis_3.5']['n'], 'bereich_mitte': br['k1_3.5_bis_5.0']['n'], 'bereich_oben': br['k1_ueber_5.0']['n'],
         'stufe_ab_minus1': K['dk_ab_-1_abwaerts']['n'], 'stufe_minus1_minus05': K['dk_-1_bis_-0.5']['n'], 'stufe_klein': K['dk_betrag_unter_0.5']['n'],
         'stufe_ab_0_5': K['dk_ab_0.5_aufwaerts_zusammengefasst']['n'], 'wh_alle': R['wiederholbarkeit']['steilheit_alle']['n'], 'wh_enden': R['wiederholbarkeit']['steilheit_primaerpaar_enden']['n'],
         'gleichgerichtet_stufe_klein': K['dk_betrag_unter_0.5']['n_gleichgerichtet_ueber_rc']}
    return Z


FEST = ('primaer', 'wh_alle', 'wh_enden')        # bereits veroeffentlichte Zahlen (Primaermenge 216, Wiederholbarkeits-Entnahmen) bleiben exakt; gerundet werden die Teilmengen


def vergroebern(Z, fest=FEST):
    """Zaehlungen 1..9 -> '<10'. Bei einem Verstoss (Differenz oder Komplement 1 bis 9) werden die beteiligten, nicht festen Zahlen auf Zehner abgerundet (Wert, True).
    Wiederholt, bis kein Verstoss bleibt."""
    g = {k: ((v, False) if not (isinstance(v, int) and 1 <= v <= 9) else ('<10', True)) for k, v in Z.items()}
    for _ in range(20):
        cur = {k: v[0] for k, v in g.items() if isinstance(v[0], int)}
        vs = [x for x in P.kleinzellen_pruefung(cur, BEZIEHUNGEN, PARTITIONEN) if x[0] != 'zaehlung']
        if not vs:
            return g
        for _typ, lab, _n in vs:
            teile = [t for t in re.split(r'-|,|sum\(|\)', lab) if t in g]
            kand = [t for t in teile if t not in fest and isinstance(g[t][0], int) and g[t][0] >= 10]
            for t in kand:
                g[t] = ((g[t][0] // 10) * 10, True)
    raise RuntimeError('Vergroeberung konvergiert nicht')


def zaehlungen_veroeffentlichen(Z):
    """Rueckgabe: ({label: (wert, gerundet)}, verbleibende Verstoesse). Eine Zaehlung 1 bis 9 wird '<10'; an einem Verstoss beteiligte Zahlen werden auf Zehner abgerundet."""
    g = vergroebern(Z)
    # Die Anzahl einflussreicher Paare ist mit dem Rest ableitbar: ist sie 1..9, ist sie '<10' und der Rest wurde schon gerundet
    n_ein = Z['n_einflussreich']
    if 1 <= n_ein <= 9 and g['ohne_einfl'][1] is False:
        g['ohne_einfl'] = ((Z['ohne_einfl'] // 10) * 10, True)
    cur = {k: v[0] for k, v in g.items() if isinstance(v[0], int)}
    rest = P.kleinzellen_pruefung({k: v for k, v in cur.items() if k != 'n_einflussreich'}, BEZIEHUNGEN, PARTITIONEN)
    return g, rest


def anzeige(g, k):
    v, gerundet = g[k]
    if v == '<10':
        return '<10'
    return f'ca. {v}' if gerundet else str(v)


# ----------------------------------------------------------------------------------------------------------------------------- Bausteine
def zeile(label, n_txt, e, G=P.G2, extra=''):
    if not e or not e.get('beurteilbar'):
        return f'| {label} | {n_txt} | nicht beurteilbar | | | {extra} |'
    k = P.B.kategorie_intervall(e, G)
    return f'| {label} | {n_txt} | {z(e["beta"])} | {iv(e["ci_lo"], e["ci_hi"])} | {kat_text(k)} | {extra} |'


def bauen():
    R = json.loads(EIN.read_text(encoding='utf-8'))
    soll = json.loads(GEBUNDEN.read_text(encoding='utf-8'))
    assert R['gate']['bestanden'] and R['gate']['wh_bestanden'], 'Gate nicht bestanden'
    Z = zaehlungen_bauen(R)
    g, rest = zaehlungen_veroeffentlichen(Z)
    assert not rest, rest
    S = R['sensitivitaeten']
    A = lambda k: anzeige(g, k)
    zeilen = []
    zeilen.append('| Schätzung | Paare | β (mV/s je mmol/L) | 95-%-Intervall | Kategorie nach Raster (G = 0,2; nicht gebunden) | Anmerkung |')
    zeilen.append('|---|---|---|---|---|---|')
    prim = R['gate']['h2']
    zeilen.append(zeile('Gebunden: Primärmodell (HF-adjustiert, Seed 255002, 2.000 Ziehungen)', A('primaer'), prim, extra='p_boot (Wert hier nicht genannt)'))
    zeilen.append(zeile('Unadjustiert (nur Achsenabschnitt und ΔK)', A('primaer'), S['unadjustiert_gleiche_faelle']))
    zeilen.append(zeile('Huber-M-Schätzer', A('primaer'), S['huber']))
    zeilen.append(zeile('Ohne einflussreiche Paare (DFBETAS-Betrag > 2/√n)', A('ohne_einfl'), S['einfluss']['ohne_einflussreiche'], extra='einflussreiche Paare: ' + ('<10' if 1 <= Z['n_einflussreich'] <= 9 else str(Z['n_einflussreich']))))
    zeilen.append(zeile('Nur Paare mit ΔHF-Betrag unter 10/min', A('abs_dhf10'), S['nur_abs_dhf_unter_10']))
    zeilen.append(zeile('Nur Kaliumanstieg', A('anstieg'), S['nur_anstieg']))
    zeilen.append(zeile('Nur Kaliumabfall', A('abfall'), S['nur_abfall']))
    zeilen.append(zeile('Eine Beobachtung je Person (erste)', A('eine_je_person') + ' Personen', S['eine_beobachtung_je_person']))
    zeilen.append(zeile('Mit Ausgangskalium als Kovariate', A('primaer'), S['mit_k1']))
    zeilen.append(zeile('Mit Δ Bikarbonat als Kovariate', A('mit_bik'), S['mit_bikarbonat']))
    gm = S['gemischtes_modell']
    zeilen.append(zeile('Gemischtes Modell (zufälliger Achsenabschnitt je Person, REML)', A('primaer'), gm, extra='Optimierer bfgs, konvergiert, Warnung "MLE an der Parametergrenze" (Personenvarianz nahe 0)'))
    zeilen.append(zeile('Ohne dokumentierte Behandlung (kein Hinweis oder unklar)', A('ohne_dok'), S['ohne_dokumentierte_behandlung']['modell']))
    zeilen.append(zeile('Alle Erstwerte (auch Erstkalium ≥ 6,0), kein Hinweis', A('alle_erst'), S['alle_erstwerte']['modell']))
    zeilen.append(zeile('Alle Kaliumquellen', A('alle_quellen'), S['alle_quellen']['modell']))
    zeilen.append(zeile('Behandlung dokumentiert (Stratum, beschreibend)', A('dokumentiert'), S['behandlung_dokumentiert']['modell']))
    br = S['ausgangsbereich']
    zeilen.append(zeile('Erstkalium 3,5 bis 5,0', A('bereich_mitte'), br['k1_3.5_bis_5.0']))
    zeilen.append(zeile('Erstkalium über 5,0 (unter 6,0)', A('bereich_oben'), br['k1_ueber_5.0']))
    zeilen.append(f'| Erstkalium bis 3,5 | {A("bereich_unten")} | nicht beurteilbar (unter 30 Paare) | | | |')
    mf = S['messfehler_kalium']['szenarien']
    zeilen.append(f'| Kaliummessfehler SD 0,1 mmol/L (λ = {z(mf[0]["lambda"], 2)}, nur Skalierung) | {A("primaer")} | {z(mf[0]["beta"])} | {iv(mf[0]["ci_lo"], mf[0]["ci_hi"])} | {kat_text(P.B.kategorie_intervall({"beurteilbar": True, "ci_lo": mf[0]["ci_lo"], "ci_hi": mf[0]["ci_hi"]}, P.G2))} | |')
    zeilen.append(f'| Kaliummessfehler SD 0,2 mmol/L (λ = {z(mf[1]["lambda"], 2)}, nur Skalierung) | {A("primaer")} | {z(mf[1]["beta"])} | {iv(mf[1]["ci_lo"], mf[1]["ci_hi"])} | {kat_text(P.B.kategorie_intervall({"beurteilbar": True, "ci_lo": mf[1]["ci_lo"], "ci_hi": mf[1]["ci_hi"]}, P.G2))} | |')
    zeilen.append(zeile('D1 gegen D3 auf denselben Paaren (Auftrag ±15 min)', A('d1'), S['d1_gleiche_paare']['d1'], extra='identisch zu D3 (per Konstruktion)'))
    tab_sens = '\n'.join(zeilen)

    sp = R['stabilitaet_je_split']
    sz = ['| Teilmenge | Paare | β | 95-%-Intervall | Kategorie (nicht gebunden) |', '|---|---|---|---|---|']
    sz.append(f'| Train | {A("train")} | {z(sp["train"]["beta"])} | {iv(sp["train"]["ci_lo"], sp["train"]["ci_hi"])} | {kat_text(P.B.kategorie_intervall(sp["train"], P.G2))} |')
    sz.append(f'| Val | {A("val")} | nicht beurteilbar (unter 30 Paare) | | |')
    sz.append(f'| Test | {A("test")} | nicht beurteilbar (unter 30 Paare) | | |')
    sz.append(f'| Val und Test zusammen (zusätzlich, nicht in der Bindung) | {A("val_test")} | {z(sp["val_und_test_zusammen"]["beta"])} | {iv(sp["val_und_test_zusammen"]["ci_lo"], sp["val_und_test_zusammen"]["ci_hi"])} | {kat_text(P.B.kategorie_intervall(sp["val_und_test_zusammen"], P.G2))} |')
    tab_split = '\n'.join(sz)

    # Seeds
    sd = R['seeds']
    vz = ['| Seed | β | untere Grenze | obere Grenze | p_boot | Kategorie |', '|---|---|---|---|---|---|']
    for i, s_ in enumerate(sd['vier']):
        vz.append(f'| {s_["seed"]}{" (gebunden)" if i == 0 else ""} | {z(s_["beta"])} | {z(s_["ci_lo"], 4)} | {z(s_["ci_hi"], 4)} | {z(s_["p_boot"])} | {kat_text(s_["kategorie"])} |')
    tab_seed = '\n'.join(vz)
    e20 = sd['b20000_gebundener_seed']
    ens = sd['ensemble_40']['zusammenfassung']
    alle = sd['alle_44_zusammenfassung']

    # Wiederholbarkeit
    wh = R['wiederholbarkeit']
    def whz(k, name, d=3):
        w = wh[k]
        return f'| {name} | {w["n"]} | {z(w["sw"], d)} | {z(w["rc"], d)} | {z(w["sw_robust"], d)} | {z(w["rc_robust"], d)} | {z(w["icc"], 3)} | {z(w["bias"], d)} |'
    tw = ['| Messgröße (Entnahmen) | Entnahmen | SD innerhalb | RC = 1,96·√2·SD | robuste SD innerhalb | robuste RC | ICC(1,1) | Bias (Fenster 1 minus 2) |', '|---|---|---|---|---|---|---|---|']
    tw.append(whz('amplitude_alle', 'T-Amplitude, mV (alle Entnahmen, gebunden reproduziert)', 4))
    tw.append(whz('steilheit_alle', 'T-Steilheit, mV/s (alle Entnahmen)', 3))
    tw.append(whz('amplitude_primaerpaar_enden', 'T-Amplitude, mV (nur Enden der Primärpaare)', 4))
    tw.append(whz('steilheit_primaerpaar_enden', 'T-Steilheit, mV/s (nur Enden der Primärpaare)', 3))
    tab_wh = '\n'.join(tw)

    kl = R['klinisch_h2']
    kl1 = R['klinisch_h1_vergleich']
    klr = R['klinisch_h2_robust_rc']
    st = kl['individuell_je_kaliumstufe']
    ts = ['| Stufe der Kaliumänderung (mmol/L) | Paare | Median ΔS (mV/s) | Median ΔS in RC-Einheiten |', '|---|---|---|---|']
    for key, name, zk in (('dk_ab_-1_abwaerts', 'ΔK ≤ −1,0', 'stufe_ab_minus1'), ('dk_-1_bis_-0.5', '−1,0 < ΔK ≤ −0,5', 'stufe_minus1_minus05'), ('dk_betrag_unter_0.5', 'ΔK-Betrag < 0,5', 'stufe_klein'),
                          ('dk_ab_0.5_aufwaerts_zusammengefasst', 'ΔK ≥ 0,5 (zusammengefasst)', 'stufe_ab_0_5')):
        v = st[key]
        ts.append(f'| {name} | {A(zk)} | {z(v["dS_median"], 2)} | {z(v["median_je_rc"], 3)} |')
    tab_stufen = '\n'.join(ts)
    anteil_klein = st['dk_betrag_unter_0.5']['anteil_gleichgerichtet_ueber_rc']

    nk1 = R['nk1_steilheit']
    nk2 = R['nk2_steilheit']
    na = R['natrium_steilheit']
    sk = R['skalenvergleich']
    sk2 = R['skalenvergleich_robuste_sw']
    sk3 = R['skalenvergleich_sw_primaerpaar_enden']
    emp = R['verhaeltnis_mcmed']

    def sk_zeile(name, k, d=3, S_=sk):
        v = S_[k]
        extra = f' ({z(v["anteil_bootstrap_ueber_null"] * 100, 0)} % der Ziehungen über 0)' if 'anteil_bootstrap_ueber_null' in v else ''
        return f'| {name} | {z(v["wert"], d)} | {iv(v["ci_lo"], v["ci_hi"], d)}{extra} |'
    ss = ['| Größe | Punktschätzer | 95-%-Intervall (gemeinsamer Clusterbootstrap) |', '|---|---|---|']
    ss.append(sk_zeile('β Amplitude (mV je mmol/L)', 'beta_T_mv_je_mmol', 4))
    ss.append(sk_zeile('β Steilheit (mV/s je mmol/L)', 'beta_S_mvs_je_mmol', 3))
    ss.append(sk_zeile('A: Amplitude in SD der Wiederholbarkeit je mmol/L', 'A_zT_je_sd_wiederholung', 3))
    ss.append(sk_zeile('A: Steilheit in SD der Wiederholbarkeit je mmol/L', 'A_zS_je_sd_wiederholung', 3))
    ss.append(sk_zeile('A: Differenz Steilheit minus Amplitude', 'A_differenz_zS_minus_zT', 3))
    ss.append(sk_zeile('B: Amplitude in SD der Residuen der Paaränderung je mmol/L', 'B_zT_je_sd_residual', 3))
    ss.append(sk_zeile('B: Steilheit in SD der Residuen der Paaränderung je mmol/L', 'B_zS_je_sd_residual', 3))
    ss.append(sk_zeile('B: Differenz Steilheit minus Amplitude', 'B_differenz_zS_minus_zT', 3))
    ss.append(sk_zeile('C: Steilheit als Amplitude (mm je mmol/L, Verhältnis 20,8/s)', 'C_S_als_amplitude_mm_je_mmol_r_spezifikation_20.8', 3))
    ss.append(sk_zeile('C: Differenz β Steilheit minus 20,8 · β Amplitude (mV/s je mmol/L)', 'C_differenz_S_minus_r_mal_T_r_spezifikation_20.8', 3))
    ss.append(sk_zeile('C: Differenz β Steilheit minus 21,4 · β Amplitude (empirisches Verhältnis MC-MED)', 'C_differenz_S_minus_r_mal_T_r_empirisch_mcmed', 3))
    tab_skala = '\n'.join(ss)

    # Zusammenfassung der Sensitivitaeten (Zaehlungen von Analysen, keine Personenzaehlungen)
    liste = [('unadjustiert', S['unadjustiert_gleiche_faelle']), ('huber', S['huber']), ('ohne_einflussreiche', S['einfluss']['ohne_einflussreiche']), ('nur_anstieg', S['nur_anstieg']), ('nur_abfall', S['nur_abfall']),
             ('mit_k1', S['mit_k1']), ('mit_bikarbonat', S['mit_bikarbonat']), ('abs_dhf_unter_10', S['nur_abs_dhf_unter_10']), ('eine_je_person', S['eine_beobachtung_je_person']),
             ('gemischtes_modell', S['gemischtes_modell']), ('ohne_dok_behandlung', S['ohne_dokumentierte_behandlung']['modell']), ('alle_erstwerte', S['alle_erstwerte']['modell']),
             ('alle_quellen', S['alle_quellen']['modell']), ('train', sp['train'])]
    ok = [(n, e) for n, e in liste if e.get('beurteilbar')]
    tally = {'analysen': len(ok), 'beta_positiv': sum(1 for _n, e in ok if e['beta'] > 0), 'untergrenze_ueber_0': sum(1 for _n, e in ok if e['ci_lo'] > 0), 'untergrenze_ueber_G': sum(1 for _n, e in ok if e['ci_lo'] >= P.G2),
             'beta_min': min(e['beta'] for _n, e in ok), 'beta_max': max(e['beta'] for _n, e in ok), 'unter_0': [n for n, e in ok if e['ci_lo'] <= 0], 'ueber_G': [n for n, e in ok if e['ci_lo'] >= P.G2]}

    oeff = {'kopf': 'post hoc, nach Sicht der Ergebnisse, explorativ, nicht gebunden', 'zaehlungen_veroeffentlicht': {k: v[0] for k, v in g.items() if k not in ('n_einflussreich',)},
            'zaehlungen_gerundet': [k for k, v in g.items() if v[1]], 'n_einflussreich': '<10' if 1 <= Z['n_einflussreich'] <= 9 else Z['n_einflussreich'], 'tally': tally,
            'tabellen': {'sensitivitaeten': tab_sens, 'split': tab_split, 'seeds': tab_seed, 'wiederholbarkeit': tab_wh, 'stufen': tab_stufen, 'skala': tab_skala}}
    OEFF.write_text(json.dumps(P.sicher(oeff), indent=1, ensure_ascii=False), encoding='utf-8')
    txt = ['# Fragmente (privat, werden in S3_POSTHOC_H2.md uebernommen)', '', '## Sensitivitaeten', tab_sens, '', '## Split', tab_split, '', '## Seeds', tab_seed, '', '## Wiederholbarkeit', tab_wh, '', '## Stufen', tab_stufen, '', '## Skala', tab_skala]
    FRAG.write_text('\n'.join(txt), encoding='utf-8')
    # Kennzahlen fuer den Text
    print('GERUNDET', oeff['zaehlungen_gerundet'], 'Einflussreiche', oeff['n_einflussreich'])
    print('TALLY', json.dumps(P.sicher(tally), ensure_ascii=False))
    print('SEEDS B20000', z(e20['ci_lo'], 4), z(e20['ci_hi'], 4), kat_text(e20['kategorie_explorativ']), 'normal', z(sd['normalnaeherung_95']['ci_lo'], 4), z(sd['normalnaeherung_95']['ci_hi'], 4))
    print('SEEDS ens40', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ens.items()})
    print('SEEDS alle44', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in alle.items()})
    print('KLINISCH H2', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kl.items() if k != 'individuell_je_kaliumstufe'})
    print('KLINISCH H2 robust', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in klr.items() if k != 'individuell_je_kaliumstufe'})
    print('KLINISCH H1', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kl1.items()})
    print('ANTEIL klein gleichgerichtet', round(anteil_klein, 3))
    print('NK1', json.dumps(P.sicher({k: v for k, v in nk1.items()}), ensure_ascii=False)[:1400])
    print('NK2', json.dumps(P.sicher(nk2), ensure_ascii=False))
    print('NA', json.dumps(P.sicher({k: nk_ for k, nk_ in na.items() if k not in ('roh', 'partiell_dk')}), ensure_ascii=False), 'roh', na['roh'].get('beta'), na['roh'].get('ci_lo'), na['roh'].get('ci_hi'), 'par', na['partiell_dk'].get('beta'), na['partiell_dk'].get('ci_lo'), na['partiell_dk'].get('ci_hi'))
    print('EMP', json.dumps(P.sicher(emp), ensure_ascii=False))
    print('SKALA robust', {k: (round(v['wert'], 3), round(v['ci_lo'], 3), round(v['ci_hi'], 3)) for k, v in sk2.items() if k.startswith('A_')})
    print('SKALA primaerpaar-enden', {k: (round(v['wert'], 3), round(v['ci_lo'], 3), round(v['ci_hi'], 3)) for k, v in sk3.items() if k.startswith('A_')})
    print('SKALA adj', json.dumps(P.sicher({k: sk[k] for k in ('dS_adjustiert_auf_dT', 'dT_adjustiert_auf_dS', 'korrelation_dT_dS', 'korrelation_bootstrap_bT_bS')}), ensure_ascii=False))
    print('HF', json.dumps(P.sicher(R['hf_abhaengigkeit']), ensure_ascii=False))
    print('Ausgabe', OEFF, FRAG)
    return oeff


def selbsttest():
    ok = True

    def check(name, cond):
        nonlocal ok
        print('  ok  ' if cond else '  FAIL', name)
        ok = ok and cond

    # Zahlen im Stil der echten Daten: Teilmenge mit Differenz 8 und einflussreiche Paare 7
    Z = {'primaer': 216, 'anstieg': 63, 'abfall': 132, 'ohne_einfl': 209, 'n_einflussreich': 7, 'mit_bik': 199, 'abs_dhf10': 142, 'eine_je_person': 208, 'na': 200, 'nk1': 118, 'nk2': 216,
         'train': 177, 'val': 19, 'test': 20, 'val_test': 39, 'd1': 146, 'ohne_dok': 240, 'alle_erst': 324, 'alle_quellen': 314, 'dokumentiert': 105, 'bereich_unten': 24, 'bereich_mitte': 139,
         'bereich_oben': 53, 'stufe_ab_minus1': 33, 'stufe_minus1_minus05': 28, 'stufe_klein': 132, 'stufe_ab_0_5': 23, 'wh_alle': 21055, 'wh_enden': 414, 'gleichgerichtet_stufe_klein': 11}
    g, rest = zaehlungen_veroeffentlichen(Z)
    check('keine Verstoesse nach der Vergroeberung', not rest)
    check('eine Beobachtung je Person (Differenz 8) ist gerundet', g['eine_je_person'][1] and g['eine_je_person'][0] == 200)
    check('Rest nach einflussreichen Paaren (Differenz 7) ist gerundet', g['ohne_einfl'][1] and g['ohne_einfl'][0] == 200)
    check('unveraenderte Zahlen bleiben exakt', g['primaer'][0] == 216 and not g['primaer'][1] and g['anstieg'][0] == 63 and not g['anstieg'][1] and g['mit_bik'][0] == 199)
    check('anzeige-Format', anzeige(g, 'eine_je_person') == 'ca. 200' and anzeige(g, 'primaer') == '216')
    # Komplement-Verstoss: Stufen 33, 28, 132, 15 -> Rest 8
    Z2 = dict(Z)
    Z2['stufe_ab_0_5'] = 15
    v = P.kleinzellen_pruefung(Z2, BEZIEHUNGEN, PARTITIONEN)
    check('Stufen mit Komplement 8 werden erkannt', any(t == 'komplement' for t, _l, _n in v))
    Z3 = dict(Z)
    Z3['test'] = 5
    g3, rest3 = zaehlungen_veroeffentlichen(Z3)
    check('Zaehlung 5 wird <10 und Komplemente werden gerundet', g3['test'][0] == '<10' and not rest3)
    check('Formatierung', z(0.4321, 3) == '0,432' and iv(0.1234, 0.5678) == '0,123 bis 0,568')
    print('Selbsttest oeffentlich:', 'bestanden' if ok else 'FEHLER')
    return ok


if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else 'bauen'
    if m == 'selbsttest':
        sys.exit(0 if selbsttest() else 1)
    bauen()
