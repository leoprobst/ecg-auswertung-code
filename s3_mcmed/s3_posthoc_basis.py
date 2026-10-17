"""POST HOC, NACH SICHT DER ERGEBNISSE, BESCHREIBEND, NICHT GEBUNDEN.

S3 MC-MED (AW-255): beschreibende Basistabelle (Auftrag K vom 17.10.2026, Schwachstellenregister SCHWACHSTELLEN_S3_v2.md, Punkte 2, 11, 35). Das Skript aendert keine
gebundene Aussage und keine gebundene Datei. Es IMPORTIERT werkzeuge/s3_mcmed/s3_auswertung_v2_2.py (Hash wird gegen die Sperrdatei geprueft, Freigabe ueber pruefe_start)
und baut mit dessen Funktionen die Tabellen (tabellen_bauen) und die Zaehlkette samt Primaerauswahl (primaer_kette). Der private Bestand wird nur lesend geoeffnet
(ATTACH READ_ONLY), der Testsplit ist enthalten (Eintrag in koordination/TESTZUGRIFFE.md vorher).

Gruppen (Personenebene, eine Zeile je Person beim ersten Paar der Gruppe; Reihenfolge ta, tb, CSN):
  A     Primaerpaare (primaer_kette, 216 Paare)
  B     gueltige konsekutive Paare (Ergebniszeit vorhanden, nicht zensiert; 5.585), die nicht in die Primaerauswahl kamen; Hauptgruende nach dem ersten
        nicht erfuellten Schritt der Zaehlkette
  C     Paare mit dokumentierter Behandlung (beh_kat = dokumentiert) unter den 5.585, unabhaengig vom EKG
  D     Paare mit Ausgangskalium >= 6,0 unter den 5.585, unabhaengig vom EKG
  C_ana Analysemenge der dokumentierten Behandlung (primaer_kette mit beh_ok = dokumentiert, Erstwert < 6,0; Tabelle S4 des Supplements)
  D_ana Analysemenge Erstwert >= 6,0 ohne Behandlungshinweis (davon_erstwert_ab_6 der Zaehlkette; Tabelle S4)
Die Gruppen B, C und D ueberschneiden sich (C und D liegen in B). A ist zu C und D disjunkt.

Modi (Python: <local path> -I):
  selbsttest   Tests an Kunstdaten (kein Zugriff auf MC-MED)
  lauf         die Rechnung auf dem privaten Bestand <local path> (liest auch den Testsplit)
Regeln: hoechstens 1 Rechenprozess, Spitze unter 1,5 GB (Waechter beendet den Prozess), keine Loeschbefehle, keine Kennungen in den Ausgaben,
Kleinzellenregel in der oeffentlichen Ausgabe auch fuer Differenzen und Komplemente.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, os, json, time, math, copy, re, threading, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
WERK = Path(str(data_path("s3.code_dir")))
sys.path.insert(0, str(WERK))
import numpy as np
import pandas as pd
import s3_auswertung_v2_2 as B
import s3_posthoc_h2 as H

OUT = Path(str(data_path("s3.posthoc_basis_dir")))
DB = B.PRIV / 's3_gesamt.duckdb'
JSON_GEBUNDEN = B.AUS / 'gesamt_maskiert.json'
CFG = B.CFG
ERST_K = CFG['primaer']['erst_k_max']
RAM_GRENZE_MB = 1450

GRUPPEN = ['A', 'B', 'C', 'D', 'C_ana', 'D_ana', 'B0']
GRUPPEN_TITEL = {'A': 'A Primärpaare', 'B': 'B gültige Paare ohne Primärauswahl', 'C': 'C dokumentierte Behandlung (alle gültigen Paare)',
                 'D': 'D Ausgangskalium ab 6,0 (alle gültigen Paare)', 'C_ana': 'C* Analysemenge dokumentierte Behandlung', 'D_ana': 'D* Analysemenge Ausgangskalium ab 6,0',
                 'B0': 'B0 (Zusatz) B ohne C und D'}
GRUENDE = ['kein_ecg_beide_enden', 'messung_ungueltig', 'polaritaet', 'ueberlappung', 'hf_fehlt', 'behandlung', 'erstwert_ab_6']


# ----------------------------------------------------------------------------------------------------------------------------- Hilfen
def sicher(o):
    return H.sicher(o)


def floor10(n):
    return (int(n) // 10) * 10


class Waechter(threading.Thread):
    """Beendet den Prozess, wenn der Arbeitsspeicher die Grenze ueberschreitet (Auftrag: unter 1,5 GB)."""

    def __init__(self, log, grenze_mb=RAM_GRENZE_MB):
        super().__init__(daemon=True)
        self.log, self.grenze, self.spitze, self.halt = log, grenze_mb, 0, False

    def run(self):
        try:
            import psutil
        except ImportError:
            return
        p = psutil.Process()
        while not self.halt:
            mb = p.memory_info().rss / 1e6
            self.spitze = max(self.spitze, mb)
            if mb > self.grenze:
                self.log('ABBRUCH: Arbeitsspeicher', int(mb), 'MB ueber Grenze', self.grenze)
                os._exit(3)
            time.sleep(0.5)


class RamReservierung:
    def __init__(self, gb, zweck, log, max_warten_s=1800):
        self.gb, self.zweck, self.log, self.rid, self.max = gb, zweck, log, None, max_warten_s

    def __enter__(self):
        t = time.time()
        while time.time() - t < self.max:
            rc, out = H.ram_aufruf('pruefen', '--gb', str(self.gb), '--seite', 'analyse')
            if rc == 0:
                rc, out = H.ram_aufruf('eintragen', '--gb', str(self.gb), '--seite', 'analyse', '--sitzung', 'S3 Basistabelle post hoc', '--pid', str(os.getpid()),
                                       '--stunden', '1', '--zweck', self.zweck)
                m = re.search(r'eingetragen (\w+)', out)
                if rc == 0 and m:
                    self.rid = m.group(1)
                    self.log('RAM reserviert', self.gb, 'GB', self.rid)
                    return self
            self.log('RAM-Reservierung wartet:', out.strip()[:120])
            time.sleep(60)
        raise RuntimeError('RAM-Reservierung nicht erhalten')

    def __exit__(self, *a):
        if self.rid:
            H.ram_aufruf('freigeben', '--id', self.rid)
            self.log('RAM freigegeben', self.rid)


# ----------------------------------------------------------------------------------------------------------------------------- Gruppen
def gruppen_bauen(Q, cfg=CFG, soll=None, log=print):
    """Gruppen A bis D_ana aus der Paartabelle Q (Ausgabe von tabellen_bauen). Zaehlkette und Primaerauswahl kommen aus B.primaer_kette. Die Zuordnung der Ausschlussgruende
    je Paar wiederholt die Filter der Kette Schritt fuer Schritt und wird gegen jede Zaehlung der gebundenen Kette (und gegen die Menge der Primaerpaare) geprueft."""
    if Q.duplicated(['CSN', 'ta', 'tb']).any():
        raise ValueError('Paarschluessel nicht eindeutig')
    prim, z, gr = B.primaer_kette(Q, cfg)
    V = Q[(~Q.ohne_ergebnis) & (~Q.zensiert)].copy()
    ecg = V.c_a.notna() & V.c_b.notna()
    gueltig = V.valid_a.eq(1) & V.valid_b.eq(1)
    pol = V.pol_gleich.astype(bool)
    kol = ~V.kollision.astype(bool)
    hf = V.hf_a.notna() & V.hf_b.notna()
    beh = V.beh_kat.eq('kein_hinweis')
    erst = V.ka < cfg['primaer']['erst_k_max']
    V['grund'] = np.select([~ecg, ~gueltig, ~pol, ~kol, ~hf, ~beh, ~erst], GRUENDE, default='primaer')
    s = ecg
    cum = {'ecg_zeitlich_beide_enden': int(s.sum())}
    s = s & gueltig
    cum['messung_gueltig_beide_enden'] = int(s.sum())
    s = s & pol
    cum['polaritaet_sicher_und_gleich'] = int(s.sum())
    s = s & kol
    cum['ohne_fensterueberlappung'] = int(s.sum())
    s = s & hf
    cum['hf_beide_enden'] = int(s.sum())
    cum['davon_behandlung_dokumentiert'] = int((s & V.beh_kat.eq('dokumentiert')).sum())
    cum['davon_behandlung_unklar'] = int((s & V.beh_kat.eq('unklar')).sum())
    s = s & beh
    cum['behandlung_zugelassen'] = int(s.sum())
    cum['davon_erstwert_ab_6'] = int((s & ~erst).sum())
    cum['primaerpaare'] = int((V.grund == 'primaer').sum())
    cum['paare_konsekutiv'] = int(len(Q))
    cum['nach_ausschluss_zensiert'] = int(len(V))
    pruef = {k: (cum[k], int(z[k])) for k in cum if k in z}
    falsch = {k: v for k, v in pruef.items() if v[0] != v[1]}
    if falsch:
        raise AssertionError(f'Zuordnung der Gruende weicht von der gebundenen Zaehlkette ab: {falsch}')
    kp = set(zip(prim.CSN, prim.ta, prim.tb))
    kv = set(zip(V.CSN[V.grund == 'primaer'], V.ta[V.grund == 'primaer'], V.tb[V.grund == 'primaer']))
    if kp != kv:
        raise AssertionError('Menge der Primaerpaare weicht von primaer_kette ab')
    if soll is not None:
        sk = soll['h1']['zaehlkette']
        gek = B.mask_tree(B.zaehlkette_vergroebern(z))
        for k, v in sk.items():
            if k == 'gerundet_auf_zehner':
                continue
            if gek.get(k) != v:
                raise AssertionError(f'Zaehlkette weicht von der gebundenen Ausgabe ab: {k} {gek.get(k)} gegen {v}')
        if B.mask_tree(gr) != soll['h1']['polaritaetsgruppen']:
            raise AssertionError('Polaritaetsgruppen weichen von der gebundenen Ausgabe ab')
    C_ana, z_c, _g = B.primaer_kette(Q, cfg, beh_ok=('dokumentiert',))
    if soll is not None and len(C_ana) != soll['h1']['behandlung_dokumentiert']['paare']:
        raise AssertionError('Analysemenge dokumentierte Behandlung weicht von der gebundenen Ausgabe ab')
    cfg_e = copy.deepcopy(cfg)
    cfg_e['primaer']['erst_k_max'] = 1e9
    pe, _ze, _g = B.primaer_kette(Q, cfg_e)
    D_ana = pe[pe.ka >= cfg['primaer']['erst_k_max']]
    if len(D_ana) != z['davon_erstwert_ab_6']:
        raise AssertionError('Analysemenge Erstwert ab 6,0 weicht von der Zaehlkette ab')
    G = {'A': V[V.grund == 'primaer'], 'B': V[V.grund != 'primaer'], 'C': V[V.beh_kat.eq('dokumentiert')], 'D': V[V.ka >= cfg['primaer']['erst_k_max']],
         'C_ana': C_ana, 'D_ana': D_ana}
    ausser = set(zip(G['C'].CSN, G['C'].ta, G['C'].tb)) | set(zip(G['D'].CSN, G['D'].ta, G['D'].tb))
    G['B0'] = G['B'][[k not in ausser for k in zip(G['B'].CSN, G['B'].ta, G['B'].tb)]]
    schl = {g: set(zip(df.CSN, df.ta, df.tb)) for g, df in G.items()}
    rel = {'A_in_B_disjunkt': not (schl['A'] & schl['B']), 'C_in_B': schl['C'] <= schl['B'], 'D_in_B': schl['D'] <= schl['B'], 'A_C_disjunkt': not (schl['A'] & schl['C']),
           'A_D_disjunkt': not (schl['A'] & schl['D']), 'C_ana_in_C': schl['C_ana'] <= schl['C'], 'D_ana_in_D': schl['D_ana'] <= schl['D'],
           'A_plus_B_gleich_V': len(G['A']) + len(G['B']) == len(V), 'B0_in_B': schl['B0'] <= schl['B'], 'B0_ohne_C_D': not (schl['B0'] & (schl['C'] | schl['D'])),
           'B0_plus_C_oder_D_gleich_B': len(G['B0']) + len(schl['C'] | schl['D']) == len(G['B'])}
    if not all(rel.values()):
        raise AssertionError(f'Mengenbeziehungen verletzt: {rel}')
    gruende = {k: int((G['B'].grund == k).sum()) for k in GRUENDE}
    if sum(gruende.values()) != len(G['B']):
        raise AssertionError('Gruende summieren sich nicht auf Gruppe B')
    info = {'zaehlkette_exakt': z, 'polaritaetsgruppen_exakt': gr, 'kumulativ_geprueft': pruef, 'gruende_b_paare': gruende, 'beziehungen': rel,
            'zaehlkette_analysemenge_dokumentiert': z_c}
    return G, info


def erste_je_person(df):
    return df.sort_values(['MRN', 'ta', 'tb', 'CSN'], kind='mergesort').drop_duplicates('MRN', keep='first')


def q3(x):
    v = pd.to_numeric(pd.Series(x), errors='coerce').dropna().to_numpy(float)
    if len(v) == 0:
        return {'n': 0}
    return {'n': int(len(v)), 'median': float(np.median(v)), 'q1': float(np.percentile(v, 25)), 'q3': float(np.percentile(v, 75)),
            'sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'min': float(v.min()), 'max': float(v.max())}


def zusammenfassung(df, ebene='personen', dispo=None):
    """Merkmale einer Gruppe. ebene personen: eine Zeile je Person beim ersten Paar der Gruppe; ebene paare: alle Paare (nur privat, zur Gegenprobe)."""
    p = erste_je_person(df) if ebene == 'personen' else df
    n = int(len(p))
    dk = pd.to_numeric(p.dk, errors='coerce')
    ok = dk.notna()
    anst, abf = ok & (dk > 1e-9), ok & (dk < -1e-9)
    out = {'paare': int(len(df)), 'einheiten': n, 'personen': int(df.MRN.nunique()),
           'alter': q3(p.age), 'geschlecht_m': {'k': int((p.sex == 1).sum()), 'n': int(p.sex.notna().sum())},
           'ka': q3(p.ka), 'dk': q3(p.dk),
           'dk_richtung': {'anstieg': int(anst.sum()), 'abfall': int(abf.sum()), 'unveraendert': int((ok & ~anst & ~abf).sum()), 'n': int(ok.sum())},
           'gap_h': q3(p.gap_h), 'hf_a': q3(p.hf_a), 'qrs_a': q3(p.qrs_ms_a) if 'qrs_ms_a' in p else {'n': 0},
           'ecg_beide_enden': {'k': int((p.c_a.notna() & p.c_b.notna()).sum()), 'n': n},
           'messung_gueltig_beide_enden': {'k': int((p.valid_a.eq(1) & p.valid_b.eq(1)).sum()), 'n': n},
           'split': {s: int((p.split == s).sum()) for s in ('train', 'val', 'test')}}
    if 'n_kl' in df:                                  # Behandlungsklassen je Paar (Fenster der Behandlungszuordnung der Bindung), immer auf Paarebene
        out['behandlung_klassen'] = {'n': int(len(df)), 'kaliumsenkend': int((df.n_kl > 0).sum()), 'calcium': int((df.n_ca > 0).sum()), 'kaliumgabe': int((df.n_kgabe > 0).sum()),
                                     'dialyse': int((df.n_dialyse > 0).sum())}
    if dispo is not None:
        d = p[['CSN']].merge(dispo, on='CSN', how='left')
        out['dispo'] = {str(k): int(v) for k, v in d.iloc[:, 1].fillna('(fehlt)').value_counts().items()}
    return out


# ----------------------------------------------------------------------------------------------------------------------------- Veroeffentlichung
def pub_zahl(n, art='abgerundet'):
    """Zaehlung fuer die oeffentliche Ausgabe: 0 und 1 bis 9 als '<10', sonst auf Zehner abgerundet (art abgerundet) oder exakt (art exakt, nur fuer Zahlen,
    die die oeffentliche Zaehlkette ohnehin nennt)."""
    n = int(n)
    if n < 10:
        return {'anzeige': '<10', 'art': 'klein'}
    if art == 'exakt':
        return {'wert': n, 'art': 'exakt'}
    return {'wert': floor10(n), 'art': 'abgerundet_10'}


def pub_anteil(k, n, n_pub, marge=10):
    """Anteil k/n in ganzen Prozent. Ist k oder das Komplement unter marge (10), wird nur die Schranke gezeigt, bezogen auf die abgerundete Gruppengroesse
    (die Schranke liegt immer bei oder ueber marge Personen). Die Pruefung der Ableitbarkeit macht s3_posthoc_basis_pruefung.py."""
    if n <= 0:
        return {'anzeige': 'n. b.', 'typ': 'nb'}
    t = math.ceil(100.0 * marge / n_pub)
    if k < marge:
        return {'anzeige': f'<{t} %', 'typ': 'klein', 'schranke': t}
    if n - k < marge:
        return {'anzeige': f'>{100 - t} %', 'typ': 'gross', 'schranke': 100 - t}
    p_ = int(round(100.0 * k / n))
    if p_ <= 0:
        return {'anzeige': '<1 %', 'typ': 'klein', 'schranke': 1}
    if p_ >= 100:
        return {'anzeige': '>99 %', 'typ': 'gross', 'schranke': 99}
    return {'anzeige': f'{p_} %', 'typ': 'prozent', 'prozent': p_}


def pub_anteil5(k, n, n_pub, marge=10):
    """Wie pub_anteil, aber auf 5 % gerundet (fuer Nenner, die exakt oeffentlich sind: dort wuerde ein ganzer Prozentsatz die Zaehlung festlegen)."""
    o = pub_anteil(k, n, n_pub, marge)
    if o['typ'] == 'prozent':
        p5 = int(5 * round(100.0 * k / n / 5))
        o = ({'anzeige': f'{p5} %', 'typ': 'prozent', 'prozent': p5, 'gerundet5': True} if p5 > 0 else {'anzeige': '<5 %', 'typ': 'klein', 'schranke': 5})
    return o


def pub_familie(ks, n, n_pub, marge=10, grob=False):
    """Vollstaendige Aufteilung (z. B. Anstieg, Abfall, unveraendert): Steht in der Aufteilung ein verdeckter Anteil, werden die anderen auf 5 % gerundet, damit der
    verdeckte Anteil nicht aus der Summe der uebrigen folgt. grob: immer auf 5 % runden (kleine Gruppen mit enger Spanne der Personenzahl, in denen ganze Prozente samt Summe
    die Zaehlungen festlegen wuerden)."""
    out = {k: pub_anteil(v, n, n_pub, marge) for k, v in ks.items()}
    if grob or any(o['typ'] in ('klein', 'gross') for o in out.values()):
        for k, o in out.items():
            if o['typ'] == 'prozent':
                p5 = int(5 * round(ks[k] / n * 100.0 / 5))
                if p5 <= 0:
                    o.update({'anzeige': '<5 %', 'typ': 'klein', 'schranke': 5})
                    o.pop('prozent', None)
                else:
                    o.update({'anzeige': f'{p5} %', 'prozent': p5, 'gerundet5': True})
    return out


def pub_stat(s, dez, ganz=False, min_n=10):
    """Median (Q1 bis Q3) und SD aus q3(); keine Angabe bei weniger als 10 Einheiten."""
    if s.get('n', 0) < min_n:
        return {'anzeige': 'nicht berichtet (<10)', 'typ': 'nb'}
    f = (lambda x: int(round(x))) if ganz else (lambda x: round(x, dez))
    return {'median': f(s['median']), 'q1': f(s['q1']), 'q3': f(s['q3']), 'sd': (round(s['sd'], max(dez, 2)) if s.get('sd') is not None else None), 'typ': 'stat'}


def oeffentlich_bauen(S, gruende, info, dispo_vorhanden):
    """S: Zusammenfassungen je Gruppe (Personenebene). Rueckgabe: oeffentliche Struktur (nur Aggregate, Kleinzellenregel angewendet)."""
    P = {'kopf': 'post hoc, nach Sicht der Ergebnisse, beschreibend, nicht gebunden', 'gruppen': {}}
    z = info['zaehlkette_exakt']
    for g in GRUPPEN:
        s = S[g]
        np_ = floor10(s['personen'])
        gr = {'titel': GRUPPEN_TITEL[g], 'paare': pub_zahl(s['paare'], 'exakt' if g in ('A', 'B', 'C_ana', 'D_ana') else 'abgerundet'), 'personen': pub_zahl(s['personen'], 'abgerundet')}
        # Spanne der moeglichen wahren Personenzahl fuer die Pruefung: Zehnerbereich, begrenzt durch die Paarzahl (Personen <= Paare)
        pa = s['paare']
        gr['personen_spanne_oeffentlich'] = [np_, min(np_ + 9, pa if g in ('A', 'B', 'C_ana', 'D_ana') else floor10(pa) + 9)]
        if g == 'A':
            gr['personen_spanne_oeffentlich'][0] = max(np_, pa - 9)      # die Zaehlkette nennt: Differenz Paare zu Personen unter 10
        m = {}
        m['alter'] = pub_stat(s['alter'], 0, ganz=True)
        m['geschlecht_m'] = pub_anteil(s['geschlecht_m']['k'], s['geschlecht_m']['n'], np_)
        m['ka'] = pub_stat(s['ka'], 1)
        m['dk'] = pub_stat(s['dk'], 2)
        r = s['dk_richtung']
        fam = pub_familie({'anstieg': r['anstieg'], 'abfall': r['abfall'], 'unveraendert': r['unveraendert']}, r['n'], np_)
        m['anstieg'], m['abfall'], m['unveraendert'] = fam['anstieg'], fam['abfall'], fam['unveraendert']
        m['gap_h'] = pub_stat(s['gap_h'], 1)
        h = s['hf_a']
        m['hf_verfuegbar'] = ({'anzeige': '100 % (Definition)', 'typ': 'definition'} if g in ('A', 'C_ana', 'D_ana') else pub_anteil(h['n'], s['einheiten'], np_))
        m['hf_a'] = pub_stat(h, 0, ganz=True)
        q = s['qrs_a']
        m['qrs_verfuegbar'] = pub_anteil(q['n'], s['einheiten'], np_)
        m['qrs_a'] = pub_stat(q, 0, ganz=True)
        if g in ('A', 'C_ana', 'D_ana'):
            m['ecg_beide_enden'] = {'anzeige': '100 % (Definition)', 'typ': 'definition'}
            m['messung_gueltig_beide_enden'] = {'anzeige': '100 % (Definition)', 'typ': 'definition'}
        else:
            e = s['ecg_beide_enden']
            m['ecg_beide_enden'] = pub_anteil(e['k'], e['n'], np_)
            e = s['messung_gueltig_beide_enden']
            m['messung_gueltig_beide_enden'] = pub_anteil(e['k'], e['n'], np_)
        if dispo_vorhanden and 'dispo' in s:
            m['dispo'] = {str(k): pub_anteil(v, s['einheiten'], np_) for k, v in s['dispo'].items()}
        gr['merkmale'] = m
        P['gruppen'][g] = gr
    P['behandlung'] = {}
    for g in ('C', 'C_ana'):
        b = S[g].get('behandlung_klassen')
        if b:
            gp = P['gruppen'][g]
            lo, hi = (gp['paare']['wert'], gp['paare']['wert']) if gp['paare']['art'] == 'exakt' else (gp['paare']['wert'], gp['paare']['wert'] + 9)
            P['behandlung'][g] = {'spanne_paare': [lo, hi], 'anteile': {k: pub_anteil5(b[k], b['n'], lo) for k in ('kaliumsenkend', 'calcium', 'kaliumgabe', 'dialyse')}}
    # Gruende der Gruppe B nach den Schritten der oeffentlichen Zaehlkette (Tabelle S1): nur Differenzen, die die Kette ohnehin nennt
    zc = {k: int(v) for k, v in z.items() if isinstance(v, (int, np.integer))}
    sch = {'kein_ecg_beide_enden': zc['nach_ausschluss_zensiert'] - zc['ecg_zeitlich_beide_enden'],
           'messung_ungueltig': zc['ecg_zeitlich_beide_enden'] - zc['messung_gueltig_beide_enden'],
           'polaritaet': zc['messung_gueltig_beide_enden'] - zc['polaritaet_sicher_und_gleich'],
           'ueberlappung_hf_behandlung': zc['polaritaet_sicher_und_gleich'] - zc['behandlung_zugelassen'],
           'erstwert_ab_6': zc['behandlung_zugelassen'] - zc['primaerpaare']}
    if sum(sch.values()) != int(S['B']['paare']):
        raise AssertionError('Gruende der Gruppe B summieren sich nicht auf die Paarzahl')
    if sch['erstwert_ab_6'] != zc['davon_erstwert_ab_6']:
        raise AssertionError('Schritt Erstwert ab 6,0 weicht ab')
    P['gruende_b'] = {k: {'wert': v, 'art': 'exakt', 'anteil_prozent': round(100.0 * v / S['B']['paare'], 1)} for k, v in sch.items()}
    P['gruende_b_einzeln_privat_hinweis'] = 'Fensterueberlappung, fehlende HF und Behandlung bleiben wie in Tabelle S1 zusammengefasst'
    return P


# ----------------------------------------------------------------------------------------------------------------------------- Markdown
def de(x, dez=0):
    s = f'{x:,.{dez}f}'
    return s.replace(',', '#').replace('.', ',').replace('#', '.')


def zelle_stat(m, dez, ganz=False, sd=False):
    if m.get('typ') == 'nb':
        return m['anzeige']
    d = 0 if ganz else dez
    t = f"{de(m['median'], d)} ({de(m['q1'], d)} bis {de(m['q3'], d)})"
    if sd:
        return f"{de(m['sd'], 2)}"
    return t


def zelle_zahl(z):
    if z['art'] == 'klein':
        return z['anzeige']
    return de(z['wert'])


def tabelle_md(P, dispo_vorhanden):
    kopf = ['Merkmal'] + [P['gruppen'][g]['titel'] for g in GRUPPEN]
    zeilen = []

    def add(name, f):
        zeilen.append([name] + [f(P['gruppen'][g]) for g in GRUPPEN])
    add('Paare (Zahl)', lambda G: zelle_zahl(G['paare']) + ('' if G['paare']['art'] == 'exakt' else ' (abgerundet)'))
    add('Personen (Zahl, auf Zehner abgerundet)', lambda G: zelle_zahl(G['personen']))
    add('Alter, Jahre: Median (Q1 bis Q3)', lambda G: zelle_stat(G['merkmale']['alter'], 0, True))
    add('Geschlecht männlich', lambda G: G['merkmale']['geschlecht_m']['anzeige'])
    add('Ausgangskalium, mmol/L: Median (Q1 bis Q3)', lambda G: zelle_stat(G['merkmale']['ka'], 1))
    add('ΔK, mmol/L: Median (Q1 bis Q3)', lambda G: zelle_stat(G['merkmale']['dk'], 2))
    add('ΔK, mmol/L: SD', lambda G: zelle_stat(G['merkmale']['dk'], 2, sd=True) if G['merkmale']['dk'].get('typ') == 'stat' else G['merkmale']['dk']['anzeige'])
    add('Kaliumanstieg (ΔK > 0)', lambda G: G['merkmale']['anstieg']['anzeige'])
    add('Kaliumabfall (ΔK < 0)', lambda G: G['merkmale']['abfall']['anzeige'])
    add('Kalium unverändert', lambda G: G['merkmale']['unveraendert']['anzeige'])
    add('Abstand der Entnahmen (Auftragszeiten), h: Median (Q1 bis Q3)', lambda G: zelle_stat(G['merkmale']['gap_h'], 1))
    add('Herzfrequenz am ersten Ende vorhanden', lambda G: G['merkmale']['hf_verfuegbar']['anzeige'])
    add('Herzfrequenz am ersten Ende, 1/min: Median (Q1 bis Q3)', lambda G: zelle_stat(G['merkmale']['hf_a'], 0, True))
    add('QRS-Dauer am ersten Ende vorhanden (Zusatz)', lambda G: G['merkmale']['qrs_verfuegbar']['anzeige'])
    add('QRS-Dauer am ersten Ende, ms: Median (Q1 bis Q3) (Zusatz)', lambda G: zelle_stat(G['merkmale']['qrs_a'], 0, True))
    add('EKG-Block an beiden Enden im Fenster', lambda G: G['merkmale']['ecg_beide_enden']['anzeige'])
    add('Gültige Messung an beiden Enden', lambda G: G['merkmale']['messung_gueltig_beide_enden']['anzeige'])
    if dispo_vorhanden:
        kats = sorted({k for g in GRUPPEN for k in P['gruppen'][g]['merkmale'].get('dispo', {})})
        for k in kats:
            add(f'Ziel der Entlassung/Aufnahme: {k}', lambda G, k=k: G['merkmale'].get('dispo', {}).get(k, {'anzeige': '<1 %'})['anzeige'])
    else:
        add('Triage, Ziel der Entlassung/Aufnahme, Aufenthaltsdauer, ED-Diagnosen', lambda G: 'nicht im Gesamtbestand; siehe Abschnitt 9')
    t = ['| ' + ' | '.join(kopf) + ' |', '|' + '---|' * len(kopf)]
    t += ['| ' + ' | '.join(z) + ' |' for z in zeilen]
    return '\n'.join(t)


def behandlung_md(P):
    nm = {'kaliumsenkend': 'kaliumsenkende Behandlung (Insulin, Glukose, Bikarbonat, Albuterol, Kaliumbinder)', 'calcium': 'Calcium intravenös', 'kaliumgabe': 'Kaliumgabe',
          'dialyse': 'Dialyseauftrag'}
    gs = [g for g in ('C', 'C_ana') if g in P['behandlung']]
    t = ['| Behandlungsklasse (Paare mit mindestens einer Gabe oder einem Auftrag der Klasse; Klassen überschneiden sich) | ' + ' | '.join(P['gruppen'][g]['titel'] for g in gs) + ' |',
         '|---|' + '---|' * len(gs)]
    for k, n in nm.items():
        t.append(f'| {n} | ' + ' | '.join(P['behandlung'][g]['anteile'][k]['anzeige'] for g in gs) + ' |')
    return '\n'.join(t)


def gruende_md(P):
    nm = {'kein_ecg_beide_enden': 'Kein EKG-Block an beiden Enden im Fenster', 'messung_ungueltig': 'EKG vorhanden, Messung an einem Ende ungültig',
          'polaritaet': 'Polarität unsicher oder verschieden', 'ueberlappung_hf_behandlung': 'Fensterüberlappung, Herzfrequenz fehlt oder dokumentierte oder unklare Behandlung (zusammengefasst wie in Tabelle S1)',
          'erstwert_ab_6': 'Ausgangskalium ab 6,0 (sonst ohne Ausschluss)'}
    t = ['| Erster nicht erfüllter Schritt der Zählkette | Paare | Anteil an B |', '|---|---|---|']
    for k, v in P['gruende_b'].items():
        t.append(f"| {nm[k]} | {de(v['wert'])} | {de(v['anteil_prozent'], 1)} % |")
    return '\n'.join(t)


# ----------------------------------------------------------------------------------------------------------------------------- Lauf
def tabellen_lesend(log):
    import duckdb
    fr = B.pruefe_start()
    log('Bindung geprueft, Freigabe', fr.hash[:12], 'Skripthash', B.sha(Path(B.__file__))[:16])
    c = duckdb.connect(':memory:')
    c.execute("SET memory_limit='500MB'; SET threads=1")
    c.execute(f"SET temp_directory='{(OUT / 'spill').as_posix()}'")
    c.execute(f"ATTACH '{DB.as_posix()}' AS s3 (READ_ONLY)")
    c.execute('USE s3')
    T = B.tabellen_bauen(c, fr, CFG)
    spalten = [r[0] for r in c.execute('DESCRIBE visit').fetchall()]
    dispo, dcol = None, [s for s in spalten if 'dispo' in s.lower()]
    if dcol:
        dispo = c.execute(f'SELECT CSN, {dcol[0]} AS dispo FROM visit').df()
    c.close()
    return T, spalten, dispo, dcol


def lauf(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    log = H.Protokoll(OUT / 'protokoll.log')
    log('Beginn lauf; Skript', B.sha(Path(__file__))[:16], 'gebundenes Skript', B.sha(Path(B.__file__))[:16], 'Datenbank (Stand)',
        dt.datetime.fromtimestamp(DB.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'), DB.stat().st_size)
    w = Waechter(log)
    w.start()
    with RamReservierung(1.5, 'S3 MC-MED Basistabelle post hoc (1 Prozess, unter 1,5 GB)', log):
        soll = json.loads(JSON_GEBUNDEN.read_text(encoding='utf-8'))
        T, spalten, dispo, dcol = tabellen_lesend(log)
        log('Tabellen gebaut; Spalten der Tabelle visit:', spalten, '| Entlassungsfeld:', dcol or 'keines', '| RSS MB', int(w.spitze))
        Q = T['Q']
        G, info = gruppen_bauen(Q, CFG, soll, log)
        log('Gates bestanden (Zaehlkette, Polaritaetsgruppen, Primaermenge, Analysemengen, Mengenbeziehungen); Gruppen (Paare):', {g: int(len(df)) for g, df in G.items()})
        S = {g: zusammenfassung(df, 'personen', dispo) for g, df in G.items()}
        Sp = {g: zusammenfassung(df, 'paare', dispo) for g, df in G.items()}
        for g in GRUPPEN:
            log(g, 'Paare', S[g]['paare'], 'Personen', S[g]['personen'], 'Alter', S[g]['alter'].get('median'), 'K1', S[g]['ka'].get('median'), 'dK', S[g]['dk'].get('median'),
                'EKG beide', S[g]['ecg_beide_enden'])
    w.halt = True
    R = {'kopf': 'post hoc, nach Sicht der Ergebnisse, beschreibend, nicht gebunden', 'erstellt': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
         'skript_sha256': B.sha(Path(__file__)), 'gebundenes_skript_sha256': B.sha(Path(B.__file__)), 'datenbank_groesse': DB.stat().st_size,
         'datenbank_stand': dt.datetime.fromtimestamp(DB.stat().st_mtime).isoformat(timespec='seconds'), 'visit_spalten': spalten, 'entlassungsfeld': dcol or None,
         'info': info, 'personenebene': S, 'paarebene_gegenprobe': Sp, 'rss_spitze_mb': int(w.spitze)}
    (OUT / 'posthoc_basis.json').write_text(json.dumps(sicher(R), indent=1, ensure_ascii=False), encoding='utf-8')
    P = oeffentlich_bauen(S, info['gruende_b_paare'], info, bool(dcol))
    P['skript_sha256'] = R['skript_sha256']
    (OUT / 'oeffentlich.json').write_text(json.dumps(sicher(P), indent=1, ensure_ascii=False), encoding='utf-8')
    (OUT / 'tabelle_fragment.md').write_text(tabelle_md(P, bool(dcol)) + '\n\n' + gruende_md(P) + '\n\n' + behandlung_md(P) + '\n', encoding='utf-8')
    log('Ausgaben geschrieben; RSS-Spitze MB', int(w.spitze), 'privat sha256 posthoc_basis.json', B.sha(OUT / 'posthoc_basis.json'))
    return R


# ----------------------------------------------------------------------------------------------------------------------------- Selbsttest
def q_kunst(n=700, seed=5):
    rng = np.random.default_rng(seed)
    Q = B._q_kunst(n, 0.02, rng)
    Q['MRN'] = np.arange(n) // 2 + 1
    Q['qrs_ms_a'] = rng.normal(95, 12, n)
    Q['age'] = rng.uniform(25, 90, n).round()
    Q['sex'] = rng.integers(0, 2, n).astype(float)
    Q['gap_h'] = rng.uniform(1, 12, n)
    Q['ka'] = rng.uniform(3.0, 5.9, n).round(1)
    Q['kb'] = (Q.ka + rng.normal(0, 0.7, n)).round(1)
    Q['dk'] = Q.kb - Q.ka
    ix = rng.permutation(n)
    Q.loc[ix[:25], 'ohne_ergebnis'] = True
    Q.loc[ix[25:60], 'zensiert'] = True
    Q.loc[ix[60:300], 'c_a'] = np.nan                       # kein EKG
    Q.loc[ix[300:350], 'valid_a'] = 0.0                     # Messung ungueltig
    Q.loc[ix[350:370], ['pol_gleich']] = False
    Q.loc[ix[370:377], 'kollision'] = True
    Q.loc[ix[377:380], 'hf_a'] = np.nan
    Q.loc[ix[380:420], 'beh_kat'] = 'dokumentiert'
    Q.loc[ix[420:435], 'beh_kat'] = 'unklar'
    Q.loc[ix[435:470], 'ka'] = 6.5
    Q.loc[ix[435:470], 'kb'] = 6.0
    Q['dk'] = Q.kb - Q.ka
    Q['pol_gleich'] = Q.pol_gleich.astype(bool)
    for c_ in ('n_kl', 'n_ca', 'n_kgabe', 'n_dialyse'):
        Q[c_] = (rng.random(n) < 0.15).astype(int)
    return Q


def selbsttest():
    import traceback
    erg = []

    def T_(name):
        def deko(f):
            try:
                f()
                erg.append((name, True, ''))
                print('  ok  ', name)
            except Exception:
                erg.append((name, False, traceback.format_exc()))
                print('  FAIL', name, '\n', traceback.format_exc())
            return f
        return deko
    h_vor = B.sha(Path(B.__file__))
    Q = q_kunst()
    G, info = gruppen_bauen(Q, CFG, None)

    @T_('Gruppen: A gleich Primaerauswahl, B plus A gleich gueltige Paare, Gruende summieren sich')
    def _():
        prim, z, _g = B.primaer_kette(Q, CFG)
        assert len(G['A']) == len(prim) == z['primaerpaare']
        assert len(G['A']) + len(G['B']) == z['nach_ausschluss_zensiert']
        assert sum(info['gruende_b_paare'].values()) == len(G['B'])
        assert all(v for v in info['beziehungen'].values())
        assert info['gruende_b_paare']['kein_ecg_beide_enden'] > 0 and info['gruende_b_paare']['ueberlappung'] > 0

    @T_('Gruppen C und D enthalten alle dokumentierten bzw. ab-6,0-Paare der gueltigen Menge, unabhaengig vom EKG')
    def _():
        V = Q[(~Q.ohne_ergebnis) & (~Q.zensiert)]
        assert len(G['C']) == int((V.beh_kat == 'dokumentiert').sum()) and len(G['D']) == int((V.ka >= 6.0).sum())
        assert len(G['C_ana']) <= len(G['C']) and len(G['D_ana']) <= len(G['D'])

    @T_('Falsche Zuordnung wird erkannt: Aenderung einer Zaehlung der Kette bricht ab')
    def _():
        Q2 = Q.copy()
        orig = B.primaer_kette

        def falsch(*a, **k):
            p, z, g = orig(*a, **k)
            z = dict(z)
            z['hf_beide_enden'] += 1
            return p, z, g
        B.primaer_kette = falsch
        try:
            try:
                gruppen_bauen(Q2, CFG, None)
            except AssertionError:
                return
            raise RuntimeError('keine Abweichung erkannt')
        finally:
            B.primaer_kette = orig

    @T_('Erste Zeile je Person: frueheste Paarzeit, ein Eintrag je Person')
    def _():
        d = pd.DataFrame({'MRN': [1, 1, 2, 2, 2], 'CSN': [5, 4, 9, 8, 7], 'ta': [30., 10., 5., 5., 7.], 'tb': [40., 20., 6., 6., 8.]})
        e = erste_je_person(d)
        assert list(e.sort_values('MRN').CSN) == [4, 8]
        assert e.MRN.is_unique

    @T_('Zusammenfassung Personenebene: Anteile, Richtung mit Toleranz, Kennzahlen')
    def _():
        d = pd.DataFrame({'MRN': np.arange(6), 'CSN': np.arange(6), 'ta': 0., 'tb': 1., 'age': [50, 60, 70, 80, 40, 30], 'sex': [1, 0, 1, 1, np.nan, 0], 'ka': 4.0,
                          'dk': [0.5, -0.5, 4.3 - 4.2 - 0.1, 0.0, 1.0, np.nan], 'gap_h': 3.0, 'hf_a': [80, np.nan, 90, 70, 60, 50], 'qrs_ms_a': np.nan,
                          'c_a': 1., 'c_b': [1., np.nan, 1., 1., 1., 1.], 'valid_a': 1., 'valid_b': 1., 'split': 'train'})
        s = zusammenfassung(d)
        assert s['dk_richtung'] == {'anstieg': 2, 'abfall': 1, 'unveraendert': 2, 'n': 5}, s['dk_richtung']
        assert s['geschlecht_m'] == {'k': 3, 'n': 5} and s['ecg_beide_enden']['k'] == 5 and s['hf_a']['n'] == 5

    @T_('Kleinzellenregel: Zaehlungen unter 10 und 0 werden nie gezeigt, Anteile mit Schranke')
    def _():
        assert pub_zahl(0)['anzeige'] == '<10' and pub_zahl(7)['anzeige'] == '<10' and pub_zahl(10)['wert'] == 10 and pub_zahl(219)['wert'] == 210
        assert pub_anteil(0, 200, 200)['anzeige'] == '<5 %' and pub_anteil(9, 200, 200)['anzeige'] == '<5 %' and pub_anteil(10, 200, 200)['anzeige'] == '5 %'
        assert pub_anteil(100, 200, 200)['anzeige'] == '50 %'
        assert pub_anteil(195, 200, 200)['anzeige'] == '>95 %' and pub_anteil(190, 200, 200)['anzeige'] == '95 %'
        assert pub_anteil(9, 4700, 4700)['anzeige'] == '<1 %' and pub_anteil(30, 4700, 4700)['anzeige'] == '1 %'
        assert pub_anteil(0, 0, 0)['typ'] == 'nb'

    @T_('Aufteilung mit verdecktem Anteil: uebrige Anteile auf 5 % gerundet')
    def _():
        f = pub_familie({'a': 53, 'b': 45, 'c': 3}, 101, 100)
        assert f['c']['anzeige'] == '<10 %' and f['a']['anzeige'] == '50 %' and f['b']['anzeige'] == '45 %', f
        g = pub_familie({'a': 40, 'b': 30, 'c': 30}, 100, 100)
        assert g['a']['anzeige'] == '40 %' and 'gerundet5' not in g['a']
        h = pub_familie({'a': 103, 'b': 1, 'c': 1}, 105, 100)
        assert h['a']['anzeige'] == '>90 %' and h['b']['anzeige'] == '<10 %'

    @T_('Gruppe B0 ist B ohne C und D')
    def _():
        assert len(G['B0']) + len(set(zip(G['C'].CSN, G['C'].ta, G['C'].tb)) | set(zip(G['D'].CSN, G['D'].ta, G['D'].tb))) == len(G['B'])

    @T_('Oeffentliche Struktur: keine Zaehlung 1 bis 9, keine Kennungen, Tabelle laesst sich bauen')
    def _():
        S = {g: zusammenfassung(df, 'personen') for g, df in G.items()}
        # Kunstdaten haben eine andere Kette als Tabelle S1; die Schritte werden nur auf Summe geprueft
        P = oeffentlich_bauen(S, info['gruende_b_paare'], info, False)
        txt = json.dumps(sicher(P), ensure_ascii=False)
        assert 'MRN' not in txt and 'CSN' not in txt
        for g in GRUPPEN:
            for k in ('paare', 'personen'):
                z = P['gruppen'][g][k]
                assert z['art'] == 'klein' or z['wert'] >= 10
        md = tabelle_md(P, False)
        assert 'Abschnitt 9' in md
        assert gruende_md(P).count('\n') == 6
        assert set(P['behandlung']) == {'C', 'C_ana'} and 'Dialyseauftrag' in behandlung_md(P)
        assert all(a['typ'] != 'prozent' or a.get('gerundet5') for g in P['behandlung'] for a in P['behandlung'][g]['anteile'].values())

    @T_('Gebundenes Skript bleibt unveraendert (Hash vor und nach dem Selbsttest)')
    def _():
        assert B.sha(Path(B.__file__)) == h_vor == '04ac15dae0f8a6a0a9b2d154e43b4ff3d240acd410d10707498fa53d48c9dcea'

    ok = sum(1 for _n, o, _t in erg if o)
    print(f'\nSelbsttest: {ok} von {len(erg)} bestanden')
    return ok == len(erg)


def main(argv):
    m = argv[1] if len(argv) > 1 else ''
    if m == 'selbsttest':
        sys.exit(0 if selbsttest() else 1)
    if m == 'lauf':
        lauf(argv)
        return
    print(__doc__)


if __name__ == '__main__':
    main(sys.argv)
