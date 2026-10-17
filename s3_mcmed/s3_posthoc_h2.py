"""POST HOC, NACH SICHT DER ERGEBNISSE, EXPLORATIV, NICHT GEBUNDEN.

S3 MC-MED (AW-255): Robustheitspruefung von H2 (T-Steilheit), Auftrag K vom 17.10.2026, nachtraeglich beschlossen, nachdem die gebundenen Ergebnisse
(<local path>) bekannt waren. Das Skript aendert keine gebundene Aussage und keine gebundene Datei: es IMPORTIERT
werkzeuge/s3_mcmed/s3_auswertung_v2_2.py (Hash wird gegen die Sperrdatei geprueft, Freigabe ueber pruefe_start) und nutzt dessen Funktionen fuer Paarbildung
(tabellen_bauen, primaer_kette), Modell (mit_hf_spline, schaetze, schaetze_huber, einfluss_pruefung, gemischtes_modell) und Bootstrap (cluster_bootstrap, ci_perz).
Jede Zahl hier ist eine explorative Beschreibung; Kategorien nach dem Raster der Spezifikation (kategorie_intervall) sind NICHT gebunden.

WIE H2 IM GEBUNDENEN SKRIPT MODELLIERT IST (h1_auswertung, Abschnitt H2; Spezifikation 6.2):
  * Population: dieselbe Primaermenge wie H1 (primaer_kette: Behandlung kein_hinweis, Erstwert < 6,0, sichere und gleiche Polaritaet, keine Fensterueberlappung,
    HF an beiden Enden), danach dieselben Zeilen wie h1_modelle (dropna auf dT, dk und die HF-Splinespalten) und zusaetzlich dropna(dS).
  * Ergebnisgroesse dS = steil_b - steil_a (mV/s), steil = -(Steilheit des absteigenden T-Schenkels) aus aw203 v2 auf dem polaritaetsgedrehten Signal.
  * Modell: dS = alpha + beta*dK + sum gamma_j*dB_j + eps, freier Achsenabschnitt, OLS; dB_j = j-te Komponente von f(HF2) - f(HF1) (eingeschraenkter kubischer Spline,
    feste Knoten 55, 70, 85, 110/min). DIE HERZFREQUENZAENDERUNG WIRD ALSO WIE BEI H1 UEBER DENSELBEN SPLINE ADJUSTIERT; ES GIBT KEINE ADJUSTIERUNG AUF dT.
    Eine unadjustierte H2-Schaetzung und die Teilmenge |dHF| < 10 gab es im gebundenen Lauf NICHT.
  * Intervall: Personen-Clusterbootstrap (Cluster MRN), 2.000 Ziehungen, Seed cfg.seeds.bootstrap + 1 = 255002, Perzentilintervall 95 %.
    Kategorie: kategorie_intervall(est, G = 0,2 mV/s je mmol/L).
  * Sensitivitaeten auf Teilmengen nutzen in der Bindung 500 Ziehungen (B4 = bootstrap_n // 4); hier ebenfalls.

Modi (Python: <local path> -I):
  selbsttest     Tests an Kunstdaten (kein Zugriff auf MC-MED)
  lauf           die explorative Rechnung auf dem privaten Bestand <local path> (liest auch den Testsplit; Eintrag in TESTZUGRIFFE.md vorher)
Regeln: hoechstens 1 Rechenprozess (RAM-Reservierung ueber ram_reservierung.py), keine Loeschbefehle, keine Kennungen in den Ausgaben, Kleinzellenregel in der
oeffentlichen Ausgabe auch fuer Differenzen und Komplemente.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, os, json, time, math, copy, pickle, subprocess, re, tempfile, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import numpy as np
import pandas as pd
import s3_auswertung_v2_2 as B

OUT = Path(str(data_path("s3.posthoc_h2_dir")))
DB = B.PRIV / 's3_gesamt.duckdb'
JSON_GEBUNDEN = B.AUS / 'gesamt_maskiert.json'
RAM = Path(str(data_path("s3.ram_reservation_py")))
CFG = B.CFG
KOV = B.hf_spalten(CFG)

SEED_GEBUNDEN = CFG['seeds']['bootstrap'] + 1                 # 255002, Seed der gebundenen H2-Schaetzung
SEEDS_ZUSATZ = [255102, 255202, 255302]                       # vor der Sicht festgelegt: drei weitere Seeds (Auftrag K)
SEEDS_ENSEMBLE = list(range(300001, 300041))                  # zusaetzlich: 40 weitere Seeds, nur zur Schaetzung der Stabilitaet der Kategorie
G2 = CFG['h2']['grenze']                                      # 0,2 mV/s je mmol/L
G1 = CFG['h1']['grenze']                                      # 0,010 mV je mmol/L
R_SPEC = 20.8                                                 # Verhaeltnis Steilheit zu Amplitude (je s), VitalDB, Spezifikation (Setzung S6)
NK_MARGE_S = CFG['nk']['grenze'] * R_SPEC                     # 0,104 mV/s je mmol/L: Nullnaehe-Marge der Steilheit = Marge der Amplitude mal 20,8 (post hoc, analog S6)
B_SENS = max(200, CFG['bootstrap_n'] // 4)                    # 500, wie in der Bindung


# ----------------------------------------------------------------------------------------------------------------------------- Hilfen
class Protokoll:
    def __init__(self, pfad=None):
        self.pfad = Path(pfad) if pfad else None
        self.t0 = time.time()

    def __call__(self, *a):
        s = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} +{time.time() - self.t0:7.1f}s " + ' '.join(str(x) for x in a)
        print(s, flush=True)
        if self.pfad:
            with open(self.pfad, 'a', encoding='utf-8') as f:
                f.write(s + '\n')


def sicher(o):
    """JSON-tauglich: numpy-Typen, NaN/Inf -> None, Pfade -> str."""
    if isinstance(o, dict):
        return {str(k): sicher(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sicher(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, Path):
        return str(o)
    return o


def kurz(e, G=G2, erwartet=+1):
    """Schaetzer knapp zusammengefasst (explorative Kategorie nach dem Raster gegen G, nicht gebunden)."""
    if not e or not e.get('beurteilbar'):
        return {'n': e.get('n') if e else None, 'beurteilbar': False}
    o = {k: e[k] for k in ('n', 'personen', 'beta', 'ci_lo', 'ci_hi', 'se_boot', 'p_boot') if k in e}
    o['beurteilbar'] = True
    o['kategorie_explorativ'] = B.kategorie_intervall(e, G, erwartet)
    return o


def pz(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 50:
        return None, None
    a, b = np.percentile(v, [lo, hi])
    return float(a), float(b)


def h2_rahmen(p, cfg=CFG):
    """Genau die Zeilen des gebundenen H2: h1_modelle (dropna dT, dk, Spline) und danach dropna(dS)."""
    q = B.mit_hf_spline(p, cfg).dropna(subset=['dT', 'dk'] + KOV)
    return q.dropna(subset=['dS'])


def h2_schaetze(q, kov=KOV, B_n=None, seed=SEED_GEBUNDEN, y='dS', x='dk'):
    return B.schaetze(q, y, x, kov, CFG, seed=seed, B=B_n)


# ----------------------------------------------------------------------------------------------------------------------------- Sensitivitaeten
def sens_h2(Q, prim, q2, Qd1=None, Qa=None, cfg=CFG, log=print):
    """Dasselbe Sensitivitaetsprogramm wie fuer H1 (h1_auswertung, d1_vergleich, alle_quellen), angewendet auf dS. Seeds: SEED_GEBUNDEN (Huber +7). B = 500."""
    S = {}
    t = time.time()

    def tick(name):
        log(f'  sens {name} fertig ({time.time() - t:.0f}s seit Start der Sensitivitaeten)')

    S['unadjustiert_gleiche_faelle'] = h2_schaetze(q2, [], B_SENS)
    tick('unadjustiert')
    S['huber'] = B.schaetze_huber(q2, 'dS', 'dk', KOV, cfg, seed=SEED_GEBUNDEN)
    tick('huber')
    S['einfluss'] = B.einfluss_pruefung(q2, 'dS', 'dk', KOV, cfg)
    tick('einfluss')
    S['ausgangsbereich'] = {
        k: B.schaetze(q2[m], 'dS', 'dk', KOV, cfg, seed=SEED_GEBUNDEN, B=B_SENS)
        for k, m in {'k1_bis_3.5': q2.ka <= 3.5, 'k1_3.5_bis_5.0': (q2.ka > 3.5) & (q2.ka <= 5.0), 'k1_ueber_5.0': q2.ka > 5.0}.items()}
    tick('ausgangsbereich')
    S['nur_anstieg'] = h2_schaetze(q2[q2.dk > 0], KOV, B_SENS)
    S['nur_abfall'] = h2_schaetze(q2[q2.dk < 0], KOV, B_SENS)
    tick('anstieg/abfall')
    S['mit_k1'] = h2_schaetze(q2, KOV + ['ka'], B_SENS)
    S['mit_bikarbonat'] = h2_schaetze(q2, KOV + ['dhco3'], B_SENS)
    tick('k1/bikarbonat')
    S['nur_abs_dhf_unter_10'] = h2_schaetze(q2[q2.dhf.abs() < 10], KOV, B_SENS)
    S['eine_beobachtung_je_person'] = h2_schaetze(q2.sort_values('ta').drop_duplicates('MRN'), KOV, B_SENS)
    tick('dhf/eine_je_person')
    S['gemischtes_modell'] = B.gemischtes_modell(q2, 'dS', 'dk', KOV, B.schaetze(q2, 'dS', 'dk', KOV, cfg, seed=SEED_GEBUNDEN, B=cfg['bootstrap_n']).get('beta'))
    tick('gemischtes_modell')
    p2, _z2, _g = B.primaer_kette(Q, cfg, beh_ok=('kein_hinweis', 'unklar'))
    S['ohne_dokumentierte_behandlung'] = {'primaerpaare': int(len(p2)), 'modell': h2_schaetze(h2_rahmen(p2, cfg), KOV, B_SENS)}
    cfg_e = copy.deepcopy(cfg)
    cfg_e['primaer']['erst_k_max'] = 1e9
    p3, _z3, _g = B.primaer_kette(Q, cfg_e)
    S['alle_erstwerte'] = {'primaerpaare': int(len(p3)), 'modell': h2_schaetze(h2_rahmen(p3, cfg), KOV, B_SENS)}
    tick('behandlung/erstwerte')
    var_dk = float(q2.dk.var())
    adj = B.schaetze(q2, 'dS', 'dk', KOV, cfg, seed=SEED_GEBUNDEN)           # Hauptschaetzung, identisch zur Bindung
    S['messfehler_kalium'] = B.messfehler_szenarien(adj, var_dk, cfg)
    if Qd1 is not None:
        p1, _z1, _g = B.primaer_kette(Qd1, cfg)
        k = p1[['CSN', 'ta', 'tb']].merge(prim[['CSN', 'ta', 'tb']], on=['CSN', 'ta', 'tb'])
        a, b = prim.merge(k, on=['CSN', 'ta', 'tb']), p1.merge(k, on=['CSN', 'ta', 'tb'])
        S['d1_gleiche_paare'] = {'primaerpaare_d1_gesamt': int(len(p1)), 'gemeinsame_paare': int(len(k)),
                                 'd3': h2_schaetze(h2_rahmen(a, cfg), KOV, B_SENS), 'd1': h2_schaetze(h2_rahmen(b, cfg), KOV, B_SENS)}
    if Qa is not None:
        pa, _za, _g = B.primaer_kette(Qa, cfg)
        S['alle_quellen'] = {'primaerpaare': int(len(pa)), 'modell': h2_schaetze(h2_rahmen(pa, cfg), KOV, B_SENS)}
    tick('d1/alle_quellen')
    pd_, _zd, _g = B.primaer_kette(Q, cfg, beh_ok=('dokumentiert',))
    S['behandlung_dokumentiert'] = {'paare': int(len(pd_)), 'modell': (h2_schaetze(h2_rahmen(pd_, cfg), KOV, B_SENS) if len(pd_) >= 100 else 'nicht_geschaetzt_n_unter_100')}
    tick('dokumentiert')
    return S


def stabilitaet_split(q2, cfg=CFG):
    """Train, Val, Test einzeln (nur schaetzbar ab 30 Paaren und 10 Personen, wie in der Bindung); zusaetzlich explorativ Val und Test zusammen."""
    out = {str(s): B.schaetze(g, 'dS', 'dk', KOV, cfg, seed=SEED_GEBUNDEN, B=B_SENS) for s, g in q2.groupby('split')}
    nt = q2[q2.split != 'train']
    out['val_und_test_zusammen'] = B.schaetze(nt, 'dS', 'dk', KOV, cfg, seed=SEED_GEBUNDEN, B=B_SENS)
    return out


def hf_abhaengigkeit(q2, cfg=CFG):
    """Wie stark haengt dS an der Herzfrequenzaenderung und wie stark ist dK mit ihr verbunden? (Beschreibung zur HF-Frage.)"""
    d = q2.dropna(subset=['dS', 'dhf', 'dk'])
    r_dk_dhf = float(np.corrcoef(d.dk, d.dhf)[0, 1])
    r_ds_dhf = float(np.corrcoef(d.dS, d.dhf)[0, 1])
    r_dt_dhf = float(np.corrcoef(d.dT, d.dhf)[0, 1]) if 'dT' in d else None
    e = B.schaetze(d, 'dS', 'dhf', ['dk'], cfg, seed=SEED_GEBUNDEN, B=B_SENS)
    return {'n': int(len(d)), 'korrelation_dk_dhf': r_dk_dhf, 'korrelation_dS_dhf': r_ds_dhf, 'korrelation_dT_dhf': r_dt_dhf, 'steigung_dS_je_dhf_bei_dk': kurz(e, G2)}


# ----------------------------------------------------------------------------------------------------------------------------- Seeds
def seed_serie(q2, seeds, B_n=None, cfg=CFG):
    zeilen = []
    for s in seeds:
        e = B.schaetze(q2, 'dS', 'dk', KOV, cfg, seed=s, B=B_n)
        zeilen.append({'seed': int(s), 'beta': e['beta'], 'ci_lo': e['ci_lo'], 'ci_hi': e['ci_hi'], 'p_boot': e['p_boot'], 'se_boot': e['se_boot'],
                       'kategorie': B.kategorie_intervall(e, G2), 'lo_ueber_G': bool(e['ci_lo'] >= G2)})
    return zeilen


def seed_zusammenfassung(zeilen):
    lo = np.array([z['ci_lo'] for z in zeilen])
    kat = {}
    for z in zeilen:
        kat[z['kategorie']] = kat.get(z['kategorie'], 0) + 1
    return {'seeds': len(zeilen), 'lo_min': float(lo.min()), 'lo_max': float(lo.max()), 'lo_mittel': float(lo.mean()), 'lo_sd': float(lo.std(ddof=1)) if len(lo) > 1 else None,
            'anteil_lo_ueber_G': float((lo >= G2).mean()), 'kategorien': kat}


# ----------------------------------------------------------------------------------------------------------------------------- Wiederholbarkeit
def wh_generisch(D, col, col_rep, ends=None):
    """Wie wh_beschreibung der Bindung (zwei Messfenster derselben Entnahme, gleiche sichere Polaritaet, keine tatsaechliche Fensterueberlappung), fuer beliebige Messgroesse.
    ends: optional Menge von (CSN, t_order), auf die eingeschraenkt wird. Zusatz: robuste SD innerhalb (1,4826*MAD der Differenzen / sqrt 2)."""
    d = D[(D.valid == 1) & (D.valid_rep == 1) & (D.pol_unsicher == 0) & (D.pol_unsicher_rep == 0) & (D.pol == D.pol_rep)]
    ueb = (d.w0_rep < d.w1_d3) & (d.w0_d3 < d.w1_rep)
    d = d[~ueb]
    if ends is not None:
        d = d[pd.MultiIndex.from_arrays([d.CSN, d.t_order]).isin(ends)]
    a, b = d[col].to_numpy(float), d[col_rep].to_numpy(float)
    e = B.icc_zweifenster(a, b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() >= 10:
        df = (a - b)[ok]
        mad = float(np.median(np.abs(df - np.median(df))))
        e['sw_robust'] = 1.4826 * mad / math.sqrt(2)
        e['rc_robust'] = 1.96 * math.sqrt(2) * e['sw_robust']
        e['median_stufe'] = float(np.median(np.concatenate([a[ok], b[ok]])))
        e['sd_zwischen_entnahmen'] = float(np.std((a[ok] + b[ok]) / 2, ddof=1))
    e['n_fensterueberlappung_ausgeschlossen'] = int(ueb.sum())
    return e


def klinisch_h2(est, wh, prim, rc_name='rc', cfg=CFG):
    """Klinische Uebersetzung der Steilheit analog zu klinisch_h1: Wiederholungsgrenze RC_S, erwartete Aenderung je mmol/L im Verhaeltnis zu RC_S, Kaliumaenderung bis RC_S
    (Gruppenmittel), individuelle Aenderung je Stufe der Kaliumaenderung. Keine individuelle Nachweisgrenze."""
    R = {'beurteilbar': bool(est.get('beurteilbar'))}
    if not est.get('beurteilbar') or not wh.get('beurteilbar'):
        return R
    rc = wh[rc_name]
    R['rc_mvs'] = rc
    R['beta_je_rc'] = est['beta'] / rc
    R['beta_je_rc_ci'] = [est['ci_lo'] / rc, est['ci_hi'] / rc]
    R['mmol_bis_rc_gruppenmittel'] = rc / est['beta'] if est['beta'] > 0 else None
    R['mmol_bis_rc_bei_ci_obergrenze'] = rc / est['ci_hi'] if est['ci_hi'] > 0 else None
    R['mmol_bis_rc_bei_ci_untergrenze'] = rc / est['ci_lo'] if est['ci_lo'] > 0 else None
    e = cfg['klinisch']['dk_strata']
    d = prim.dropna(subset=['dS', 'dk'])
    st = {'dk_ab_-1_abwaerts': d.dk <= e[0], 'dk_-1_bis_-0.5': (d.dk > e[0]) & (d.dk <= e[1]), 'dk_betrag_unter_0.5': (d.dk > e[1]) & (d.dk < e[2]),
          'dk_0.5_bis_1': (d.dk >= e[2]) & (d.dk < e[3]), 'dk_ab_1_aufwaerts': d.dk >= e[3]}
    # zusaetzlich fuer die oeffentliche Ausgabe: die beiden obersten Stufen zusammengefasst (die oberste Stufe allein hat unter 10 Paare und waere als Komplement ableitbar)
    st['dk_ab_0.5_aufwaerts_zusammengefasst'] = d.dk >= e[2]
    tab = {}
    for k, m in st.items():
        g = d[m]
        o = {'n': int(len(g))}
        if len(g) >= 10:
            o['dS_median'] = float(g.dS.median())
            o['dS_iqr'] = [float(g.dS.quantile(.25)), float(g.dS.quantile(.75))]
            o['median_je_rc'] = float(g.dS.median() / rc)
            n_ueber = int((g.dS.abs() > rc).sum())
            n_gleich = int(((np.sign(g.dS) == np.sign(g.dk)) & (g.dS.abs() > rc)).sum())
            o['n_betrag_ueber_rc'], o['n_gleichgerichtet_ueber_rc'] = n_ueber, n_gleich
            o['anteil_betrag_ueber_rc'] = n_ueber / len(g)
            o['anteil_gleichgerichtet_ueber_rc'] = n_gleich / len(g)
        tab[k] = o
    R['individuell_je_kaliumstufe'] = tab
    return R


# ----------------------------------------------------------------------------------------------------------------------------- Kontrollen (NK1/NK2/Natrium) fuer die Steilheit
def nk1_steil(prim, Qsh, cfg=CFG, B_n=None):
    """NK1-Analog: gleiche Paarlogik wie nk1_auswertung (Primaerpaare, beide zeitversetzten Messungen gueltig, sicher, gleichpolig, ohne Fensterueberlappung, HF vorhanden),
    Ergebnisgroesse dS statt dT. Zusatz: gleiche Paare mit Amplitude, und Kontrast (primaere Fenster minus versetzte Fenster) mit Clusterbootstrap."""
    if Qsh is None or not len(Qsh):
        return {'n': 0, 'kategorie': 'nicht_beurteilbar'}
    keys = prim[['CSN', 'ta', 'tb']]
    q = Qsh.merge(keys, on=['CSN', 'ta', 'tb'])
    q = q[q.valid_a.eq(1) & q.valid_b.eq(1) & q.pol_gleich & ~q.kollision & q.hf_a.notna() & q.hf_b.notna()]
    if len(q) < cfg['nk']['n_min']:
        return {'n': int(len(q)), 'kategorie': 'nicht_beurteilbar'}
    q = B.mit_hf_spline(q, cfg)
    R = {'n_paare': int(len(q))}
    for y, marge, G, name in (('dS', NK_MARGE_S, G2, 'steilheit'), ('dT', cfg['nk']['grenze'], G1, 'amplitude')):
        d = q.dropna(subset=[y, 'dk'] + KOV)
        e = B.schaetze(d, y, 'dk', KOV, cfg, seed=cfg['seeds']['nk1'], B=B_n or max(200, cfg['bootstrap_n'] // 2))
        if e.get('beurteilbar'):
            e['halbbreite'] = (e['ci_hi'] - e['ci_lo']) / 2
            e['kategorie_nullnaehe'] = B.kategorie_nullnaehe(e, marge)
            e['marge'] = marge
        R[name] = e
    # Kontrast auf denselben Paaren: primaere Fenster (dS_p, dT_p) minus versetzte Fenster
    pp = prim[['CSN', 'ta', 'tb', 'dS', 'dT']].rename(columns={'dS': 'dS_p', 'dT': 'dT_p'})
    c = q.merge(pp, on=['CSN', 'ta', 'tb'], how='left').dropna(subset=['dS', 'dT', 'dS_p', 'dT_p', 'dk'] + KOV)
    R['n_kontrast'] = int(len(c))
    if len(c) >= cfg['min_schaetzung']['paare']:
        def stat(t):
            return np.array([B.steigung(t, 'dS_p', 'dk', KOV), B.steigung(t, 'dS', 'dk', KOV), B.steigung(t, 'dT_p', 'dk', KOV), B.steigung(t, 'dT', 'dk', KOV)])
        est = stat(c)
        boot = B.cluster_bootstrap(c, 'MRN', stat, B_n or max(200, cfg['bootstrap_n'] // 2), cfg['seeds']['nk1'] + 100)
        for j, nm in ((0, 'steil_primaer'), (1, 'steil_versetzt'), (2, 'amp_primaer'), (3, 'amp_versetzt')):
            R.setdefault('gleiche_paare', {})[nm] = {'beta': float(est[j])}
        for nm, a, b in (('steilheit', 0, 1), ('amplitude', 2, 3)):
            lo, hi = pz(boot[:, a] - boot[:, b])
            R[f'kontrast_{nm}'] = {'differenz': float(est[a] - est[b]), 'ci_lo': lo, 'ci_hi': hi}
    return R


def nk2_steil(prim, cfg=CFG):
    """NK2-Analog: (dS, HF-Splinedifferenz) einer Paarzeile einer anderen Person zugeordnet (feste Vollstichprobe, wie nk2_auswertung)."""
    q = B.mit_hf_spline(prim, cfg).dropna(subset=['dS', 'dk'] + KOV).reset_index(drop=True)
    perms = B.nk2_permutationen(len(q), q.MRN.to_numpy(), cfg['nk2_n'], cfg['seeds']['nk2']) if len(q) >= cfg['min_schaetzung']['paare'] else None
    if perms is None:
        return {'n': int(len(q)), 'kategorie': 'nicht_beurteilbar'}
    obs = float(B.steigung(q, 'dS', 'dk', KOV))
    out = []
    for p in perms:
        t = q.copy()
        for col in ['dS'] + KOV:
            t[col] = q[col].to_numpy()[p]
        out.append(float(B.steigung(t, 'dS', 'dk', KOV)))
    out = np.array(out)
    est = {'beurteilbar': True, 'ci_lo': float(np.percentile(out, 2.5)), 'ci_hi': float(np.percentile(out, 97.5))}
    return {'n': int(len(q)), 'beta_beobachtet': obs, 'null_mittel': float(out.mean()), 'null_sd': float(out.std(ddof=1)), 'q025': est['ci_lo'], 'q975': est['ci_hi'],
            'p_perm': float(((np.abs(out) >= abs(obs)).sum() + 1) / (len(out) + 1)), 'kategorie_nullnaehe': B.kategorie_nullnaehe(est, NK_MARGE_S), 'marge': NK_MARGE_S}


def na_steil(prim, cfg=CFG):
    """Natrium-Analog: dS je 5 mmol/L Natrium, HF-adjustiert; bei |r(dNa, dK)| > 0,30 partiell auf dK (wie na_kontrolle)."""
    q = B.mit_hf_spline(prim, cfg).dropna(subset=['dS', 'dna', 'dk'] + KOV).copy()
    if len(q) < cfg['min_schaetzung']['paare']:
        return {'n': int(len(q)), 'kategorie': 'nicht_beurteilbar'}
    q['dna5'] = q.dna / cfg['na']['schritt']
    roh = B.schaetze(q, 'dS', 'dna5', KOV, cfg, seed=cfg['seeds']['bootstrap'] + 3)
    par = B.schaetze(q, 'dS', 'dna5', KOV + ['dk'], cfg, seed=cfg['seeds']['bootstrap'] + 4)
    r = float(np.corrcoef(q.dna, q.dk)[0, 1])
    rein = abs(r) <= cfg['na']['korr_max']
    mass = roh if rein else par
    return {'n': int(len(q)), 'roh': kurz(roh), 'partiell_dk': kurz(par), 'korrelation_dna_dk': r, 'kontrolle_rein': bool(rein), 'massgebend': 'roh' if rein else 'partiell_dk',
            'kategorie_nullnaehe': B.kategorie_nullnaehe(mass, NK_MARGE_S), 'marge': NK_MARGE_S}


# ----------------------------------------------------------------------------------------------------------------------------- Skalenvergleich H1 gegen H2
def empirisches_verhaeltnis(D):
    """Verhaeltnis Steilheit zu Amplitude in MC-MED (je s): Median von steil/t_amp ueber gueltige Messungen mit t_amp >= 0,10 mV, und Verhaeltnis robuster Streuungen."""
    d = D[(D.valid == 1) & (D.pol_unsicher == 0)].dropna(subset=['t_amp', 'steil'])
    d = d[d.t_amp >= 0.10]
    if len(d) < 100:
        return {'beurteilbar': False}
    ratio = (d.steil / d.t_amp).to_numpy()
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    return {'beurteilbar': True, 'n_messungen': int(len(d)), 'median_verhaeltnis': float(np.median(ratio)), 'iqr_verhaeltnis': [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
            'verhaeltnis_robuste_sd': float(mad(d.steil.to_numpy()) / mad(d.t_amp.to_numpy())), 'korrelation_steil_amp': float(np.corrcoef(d.steil, d.t_amp)[0, 1])}


def skalenvergleich(q, sw_t, sw_s, r_emp, cfg=CFG, B_n=None, seed=SEED_GEBUNDEN):
    """H1 und H2 auf derselben Skala. q: Zeilen mit dT, dS, dk, HF-Spline (gleiche Faelle fuer beide). Gemeinsamer Clusterbootstrap (gleiche Ziehungen) fuer beide Steigungen.
    (A) Standardisierung je SD der Wiederholbarkeit (sw): zT = bT/sw_t, zS = bS/sw_s; Differenz zS - zT.
    (B) Standardisierung je Residual-SD der Paaraenderung (sd_res).
    (C) Amplitudenaequivalent der Steilheit: bS/r (mV je mmol/L); Steilheitsaequivalent der Amplitude: r*bT; Differenz bS - r*bT mit r = 20,8 (Spezifikation) und r_emp (MC-MED).
    Die Nenner (sw, sd_res, r) werden als Konstanten behandelt (Nenner aus 21.000 Entnahmen bzw. dem Datensatz selbst; Unsicherheit nicht fortgepflanzt)."""
    q = q.dropna(subset=['dT', 'dS', 'dk'] + KOV)

    def res_sd(y):
        X = np.column_stack([np.ones(len(q)), q.dk.to_numpy()] + [q[k].to_numpy() for k in KOV])
        r = q[y].to_numpy() - X @ B.ols(X, q[y].to_numpy())
        return float(r.std(ddof=X.shape[1]))

    sdr_t, sdr_s = res_sd('dT'), res_sd('dS')
    est = np.array([B.steigung(q, 'dT', 'dk', KOV), B.steigung(q, 'dS', 'dk', KOV)])
    boot = B.cluster_bootstrap(q, 'MRN', lambda t: np.array([B.steigung(t, 'dT', 'dk', KOV), B.steigung(t, 'dS', 'dk', KOV)]), B_n or CFG['bootstrap_n'], seed)
    bt, bs = boot[:, 0], boot[:, 1]
    r_list = [('r_spezifikation_20.8', R_SPEC)] + ([('r_empirisch_mcmed', r_emp)] if r_emp else [])
    maße = {
        'beta_T_mv_je_mmol': (est[0], bt),
        'beta_S_mvs_je_mmol': (est[1], bs),
        'A_zT_je_sd_wiederholung': (est[0] / sw_t, bt / sw_t),
        'A_zS_je_sd_wiederholung': (est[1] / sw_s, bs / sw_s),
        'A_differenz_zS_minus_zT': (est[1] / sw_s - est[0] / sw_t, bs / sw_s - bt / sw_t),
        'B_zT_je_sd_residual': (est[0] / sdr_t, bt / sdr_t),
        'B_zS_je_sd_residual': (est[1] / sdr_s, bs / sdr_s),
        'B_differenz_zS_minus_zT': (est[1] / sdr_s - est[0] / sdr_t, bs / sdr_s - bt / sdr_t),
    }
    for nm, r in r_list:
        maße[f'C_S_als_amplitude_mv_je_mmol_{nm}'] = (est[1] / r, bs / r)
        maße[f'C_S_als_amplitude_mm_je_mmol_{nm}'] = (est[1] / r * 10, bs / r * 10)
        maße[f'C_T_als_steilheit_mvs_je_mmol_{nm}'] = (est[0] * r, bt * r)
        maße[f'C_differenz_S_minus_r_mal_T_{nm}'] = (est[1] - r * est[0], bs - r * bt)
    out = {}
    for k, (pt, arr) in maße.items():
        lo, hi = pz(arr)
        o = {'wert': float(pt), 'ci_lo': lo, 'ci_hi': hi}
        if 'differenz' in k:
            o['anteil_bootstrap_ueber_null'] = float((arr > 0).mean())
        out[k] = o
    out['nenner'] = {'sw_T': sw_t, 'sw_S': sw_s, 'sd_res_T': sdr_t, 'sd_res_S': sdr_s, 'r_spezifikation': R_SPEC, 'r_empirisch': r_emp}
    out['n'] = int(len(q))
    out['korrelation_bootstrap_bT_bS'] = float(np.corrcoef(bt, bs)[0, 1])
    # Steilheit zusaetzlich zur Amplitude: dS ~ dK + HF + dT (und umgekehrt dT ~ dK + HF + dS), Hinweis: dT ist selbst eine Folge von dK; Messfehler in dT schwaecht die Adjustierung ab
    out['dS_adjustiert_auf_dT'] = kurz(B.schaetze(q, 'dS', 'dk', KOV + ['dT'], cfg, seed=seed, B=B_n or CFG['bootstrap_n']), G2)
    out['dT_adjustiert_auf_dS'] = kurz(B.schaetze(q, 'dT', 'dk', KOV + ['dS'], cfg, seed=seed, B=B_n or CFG['bootstrap_n']), G1)
    out['korrelation_dT_dS'] = float(np.corrcoef(q.dT, q.dS)[0, 1])
    return out


# ----------------------------------------------------------------------------------------------------------------------------- Kleinzellen
def kleinzellen_pruefung(zaehlungen, beziehungen, partitionen=()):
    """zaehlungen: {label: n}; beziehungen: [(kind_label, eltern_label)] (Teilmenge); partitionen: [(eltern_label, [kinder_labels])] (Komplement muss 0 oder >= 10 sein).
    Rueckgabe: Liste der Verstoesse. Verstoss: eine Zaehlung 1..9, ein Differenzbetrag Eltern - Kind in 1..9, ein Komplement einer Partition in 1..9."""
    v = []
    for k, n in zaehlungen.items():
        if isinstance(n, (int, np.integer)) and 1 <= n <= 9:
            v.append(('zaehlung', k, int(n)))
    for kind, eltern in beziehungen:
        if kind in zaehlungen and eltern in zaehlungen:
            d = zaehlungen[eltern] - zaehlungen[kind]
            if 1 <= d <= 9:
                v.append(('differenz', f'{eltern}-{kind}', int(d)))
    for eltern, kinder in partitionen:
        if eltern in zaehlungen and all(k in zaehlungen for k in kinder):
            d = zaehlungen[eltern] - sum(zaehlungen[k] for k in kinder)
            if 1 <= d <= 9:
                v.append(('komplement', f'{eltern}-sum({",".join(kinder)})', int(d)))
    return v


def vergroebern_paarweise(zaehlungen, beziehungen, partitionen=()):
    """Gibt Zaehlungen zurueck, in denen jede an einem Verstoss beteiligte Zahl auf Zehner abgerundet ist (Wert, True). Zaehlungen 1..9 -> '<10'. Wiederholt bis kein Verstoss bleibt."""
    z = {k: (v, False) for k, v in zaehlungen.items()}
    for _ in range(10):
        cur = {k: v[0] for k, v in z.items() if isinstance(v[0], (int, np.integer))}
        verstoesse = kleinzellen_pruefung(cur, beziehungen, partitionen)
        if not verstoesse:
            break
        for typ, lab, _n in verstoesse:
            if typ == 'zaehlung':
                z[lab] = ('<10', True)
                continue
            teile = re.split(r'-|,|sum\(|\)', lab)
            for t_ in teile:
                if t_ in z and isinstance(z[t_][0], (int, np.integer)) and z[t_][0] >= 10:
                    z[t_] = ((int(z[t_][0]) // 10) * 10, True)
    return z


# ----------------------------------------------------------------------------------------------------------------------------- Ablauf
def ram_aufruf(*args):
    r = subprocess.run([sys.executable, '-I', str(RAM), *args], capture_output=True)
    return r.returncode, r.stdout.decode('cp850', errors='replace') + r.stderr.decode('cp850', errors='replace')


class RamReservierung:
    def __init__(self, gb, zweck, log, max_warten_s=1800):
        self.gb, self.zweck, self.log, self.rid, self.max = gb, zweck, log, None, max_warten_s

    def __enter__(self):
        t = time.time()
        while time.time() - t < self.max:
            rc, out = ram_aufruf('pruefen', '--gb', str(self.gb), '--seite', 'analyse')
            if rc == 0:
                rc, out = ram_aufruf('eintragen', '--gb', str(self.gb), '--seite', 'analyse', '--sitzung', 'S3 post hoc H2', '--pid', str(os.getpid()), '--stunden', '3', '--zweck', self.zweck)
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
            ram_aufruf('freigeben', '--id', self.rid)
            self.log('RAM freigegeben', self.rid)


def verbinden_lesend(pfad):
    """Nur lesend: ':memory:' plus ATTACH READ_ONLY, damit TEMP-Tabellen der gebundenen Funktionen moeglich sind, ohne den Bestand zu veraendern."""
    import duckdb
    c = duckdb.connect(':memory:')
    c.execute("SET memory_limit='700MB'; SET threads=2")
    c.execute(f"ATTACH '{Path(pfad).as_posix()}' AS s3 (READ_ONLY)")
    c.execute('USE s3')
    return c


def tabellen_laden(log, cache=None):
    """Tabellen wie der gebundene Lauf (tabellen_bauen unter der Freigabe der Bindung). Optional als Pickle im privaten Ordner zwischengespeichert."""
    if cache and Path(cache).exists():
        log('Tabellen aus Zwischenspeicher', cache)
        with open(cache, 'rb') as f:
            return pickle.load(f)
    fr = B.pruefe_start()
    log('Bindung geprueft, Freigabe', fr.hash[:12], 'Skripthash', B.sha(Path(B.__file__))[:16])
    c = verbinden_lesend(DB)
    T = B.tabellen_bauen(c, fr, CFG)
    c.close()
    if cache:
        with open(cache, 'wb') as f:
            pickle.dump(T, f, protocol=4)
    return T


def analyse(T, log, ensemble=True, soll=None):
    """Gesamte explorative Rechnung. soll: gebundene JSON (Gate), None bei Kunstdaten."""
    t0 = time.time()
    R = {'kopf': 'post hoc, nach Sicht der Ergebnisse, explorativ, nicht gebunden', 'erstellt': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
         'modell_h2': {'population': 'Primaermenge wie H1 (primaer_kette), Zeilen wie h1_modelle plus dropna(dS)', 'ergebnisgroesse': 'dS = steil_b - steil_a (mV/s)',
                       'hf': 'HF-Aenderung adjustiert ueber f(HF2)-f(HF1), eingeschraenkter kubischer Spline, Knoten 55/70/85/110; keine Adjustierung auf dT; unadjustiert und |dHF|<10 im gebundenen Lauf nicht gerechnet',
                       'bootstrap': 'Personen-Clusterbootstrap MRN, 2000 Ziehungen, Seed 255002, Perzentil 95 %', 'grenze_G': G2},
         'seeds_festgelegt': {'gebunden': SEED_GEBUNDEN, 'zusatz': SEEDS_ZUSATZ, 'ensemble': [SEEDS_ENSEMBLE[0], SEEDS_ENSEMBLE[-1]]}}
    Q = T['Q']
    prim, z, gruppen = B.primaer_kette(Q, CFG)
    q2 = h2_rahmen(prim, CFG)
    q1 = B.mit_hf_spline(prim, CFG).dropna(subset=['dT', 'dk'] + KOV)
    R['rahmen'] = {'primaerpaare': int(len(prim)), 'zeilen_h1': int(len(q1)), 'zeilen_h2': int(len(q2))}
    log('Primaermenge', len(prim), 'Zeilen H1', len(q1), 'Zeilen H2', len(q2))

    # --- Gate: Reproduktion der gebundenen Zahlen
    adj1 = B.schaetze(q1, 'dT', 'dk', KOV, CFG)                        # H1, Seed 255001, B = 2000
    adj2 = B.schaetze(q2, 'dS', 'dk', KOV, CFG, seed=SEED_GEBUNDEN)    # H2, Seed 255002, B = 2000
    R['gate'] = {'h1': kurz(adj1, G1), 'h2': kurz(adj2, G2)}
    if soll is not None:
        g = {'h1_beta': (adj1['beta'], soll['h1']['primaer']['beta']), 'h1_lo': (adj1['ci_lo'], soll['h1']['primaer']['ci_lo']), 'h1_hi': (adj1['ci_hi'], soll['h1']['primaer']['ci_hi']),
             'h2_beta': (adj2['beta'], soll['h1']['h2_steilheit']['beta']), 'h2_lo': (adj2['ci_lo'], soll['h1']['h2_steilheit']['ci_lo']), 'h2_hi': (adj2['ci_hi'], soll['h1']['h2_steilheit']['ci_hi']),
             'h2_p': (adj2['p_boot'], soll['h1']['h2_steilheit']['p_boot'])}
        R['gate']['vergleich'] = {k: {'neu': a, 'gebunden': b, 'abweichung': abs(a - b)} for k, (a, b) in g.items()}
        R['gate']['n'] = {'h1': (adj1['n'], soll['h1']['primaer']['n']), 'h2': (adj2['n'], soll['h1']['h2_steilheit']['n'])}
        R['gate']['bestanden'] = bool(all(v['abweichung'] < 1e-9 for v in R['gate']['vergleich'].values()) and adj1['n'] == soll['h1']['primaer']['n'] and adj2['n'] == soll['h1']['h2_steilheit']['n'])
        log('GATE', 'bestanden' if R['gate']['bestanden'] else 'NICHT BESTANDEN', {k: round(v['abweichung'], 12) for k, v in R['gate']['vergleich'].items()})
        if not R['gate']['bestanden']:
            R['abbruch'] = 'Gate nicht bestanden: Reproduktion der gebundenen H1/H2-Zahlen weicht ab'
            return R

    # --- Seeds
    log('Seeds: gebunden + 3 weitere')
    ser = seed_serie(q2, [SEED_GEBUNDEN] + SEEDS_ZUSATZ)
    R['seeds'] = {'vier': ser, 'zusammenfassung_vier': seed_zusammenfassung(ser)}
    log('  untere Grenzen', [round(s['ci_lo'], 4) for s in ser], [s['kategorie'] for s in ser])
    e20 = B.schaetze(q2, 'dS', 'dk', KOV, CFG, seed=SEED_GEBUNDEN, B=CFG['bootstrap_n'] * 10)
    R['seeds']['b20000_gebundener_seed'] = kurz(e20, G2)
    R['seeds']['normalnaeherung_95'] = {'ci_lo': adj2['beta'] - 1.96 * adj2['se_boot'], 'ci_hi': adj2['beta'] + 1.96 * adj2['se_boot'], 'se_boot': adj2['se_boot']}
    log('  B=20000:', e20['ci_lo'], e20['ci_hi'])
    if ensemble:
        ens = seed_serie(q2, SEEDS_ENSEMBLE)
        R['seeds']['ensemble_40'] = {'zeilen': ens, 'zusammenfassung': seed_zusammenfassung(ens)}
        alle = ser + ens
        R['seeds']['alle_44_zusammenfassung'] = seed_zusammenfassung(alle)
        log('  Ensemble:', R['seeds']['alle_44_zusammenfassung'])
    # Abstand der unteren Grenze zu G in Standardfehlern des Quantils (Normalnaeherung): wie wahrscheinlich ein anderer Seed unter G faellt
    # (nur Beschreibung; die empirische Verteilung ueber die Seeds ist maßgebend)
    # --- Sensitivitaeten
    log('Sensitivitaeten H2')
    S = sens_h2(Q, prim, q2, T.get('Qd1'), T.get('Qa'), CFG, log)
    R['sensitivitaeten'] = S
    R['stabilitaet_je_split'] = stabilitaet_split(q2)
    R['hf_abhaengigkeit'] = hf_abhaengigkeit(q2)
    log('Stabilitaet je Split fertig')

    # --- Wiederholbarkeit der Steilheit
    D = T['Dm']
    ends = set(zip(prim.CSN, prim.ta)) | set(zip(prim.CSN, prim.tb))
    wh_amp = wh_generisch(D, 't_amp', 't_amp_rep')
    wh_s = wh_generisch(D, 'steil', 'steil_rep')
    wh_s_prim = wh_generisch(D, 'steil', 'steil_rep', ends)
    wh_amp_prim = wh_generisch(D, 't_amp', 't_amp_rep', ends)
    R['wiederholbarkeit'] = {'amplitude_alle': wh_amp, 'steilheit_alle': wh_s, 'amplitude_primaerpaar_enden': wh_amp_prim, 'steilheit_primaerpaar_enden': wh_s_prim}
    if soll is not None:
        bw = soll['beschreibungen']['wiederholbarkeit']
        R['gate']['wh_amplitude'] = {'n': (wh_amp['n'], bw['n']), 'sw': (wh_amp['sw'], bw['sw']), 'rc': (wh_amp['rc'], bw['rc'])}
        R['gate']['wh_bestanden'] = bool(wh_amp['n'] == bw['n'] and abs(wh_amp['sw'] - bw['sw']) < 1e-9 and abs(wh_amp['rc'] - bw['rc']) < 1e-9)
        log('GATE Wiederholbarkeit Amplitude', 'bestanden' if R['gate']['wh_bestanden'] else 'NICHT BESTANDEN')
    log('Wiederholbarkeit', {k: (v.get('n'), v.get('sw'), v.get('icc'), v.get('rc')) for k, v in R['wiederholbarkeit'].items()})
    R['klinisch_h2'] = klinisch_h2(adj2, wh_s, q2)
    R['klinisch_h2_primaerpaar_enden'] = klinisch_h2(adj2, wh_s_prim, q2)
    R['klinisch_h2_robust_rc'] = klinisch_h2(adj2, wh_s, q2, rc_name='rc_robust') if 'rc_robust' in wh_s else None
    # Amplitude in derselben Form (zum Vergleich, aus den gebundenen Zahlen reproduziert)
    R['klinisch_h1_vergleich'] = {'rc_mv': wh_amp['rc'], 'beta_je_rc': adj1['beta'] / wh_amp['rc'], 'beta_je_rc_ci': [adj1['ci_lo'] / wh_amp['rc'], adj1['ci_hi'] / wh_amp['rc']],
                                  'mmol_bis_rc_gruppenmittel': (wh_amp['rc'] / adj1['beta']) if adj1['beta'] > 0 else None,
                                  'mmol_bis_rc_bei_ci_obergrenze': (wh_amp['rc'] / adj1['ci_hi']) if adj1['ci_hi'] > 0 else None}

    # --- NK1/NK2/Natrium fuer die Steilheit
    log('Kontrollen')
    R['nk1_steilheit'] = nk1_steil(prim, T.get('Qsh'), CFG)
    R['nk2_steilheit'] = nk2_steil(prim, CFG)
    R['natrium_steilheit'] = na_steil(prim, CFG)

    # --- Skalenvergleich
    log('Skalenvergleich')
    emp = empirisches_verhaeltnis(D)
    R['verhaeltnis_mcmed'] = emp
    r_emp = emp.get('median_verhaeltnis') if emp.get('beurteilbar') else None
    R['skalenvergleich'] = skalenvergleich(q2, wh_amp['sw'], wh_s['sw'], r_emp, CFG)
    R['skalenvergleich_robuste_sw'] = skalenvergleich(q2, wh_amp.get('sw_robust', wh_amp['sw']), wh_s.get('sw_robust', wh_s['sw']), r_emp, CFG) if 'sw_robust' in wh_s else None
    R['skalenvergleich_sw_primaerpaar_enden'] = (skalenvergleich(q2, wh_amp_prim['sw'], wh_s_prim['sw'], r_emp, CFG)
                                                 if wh_amp_prim.get('beurteilbar') and wh_s_prim.get('beurteilbar') else None)
    R['dauer_s'] = time.time() - t0
    return R


# ----------------------------------------------------------------------------------------------------------------------------- Selbsttest (Kunstdaten)
def selbsttest():
    import traceback
    ergebnisse = []

    def T_(name):
        def deko(f):
            try:
                f()
                ergebnisse.append((name, True, ''))
                print('  ok  ', name)
            except Exception as e:
                ergebnisse.append((name, False, traceback.format_exc()))
                print('  FAIL', name, '\n', traceback.format_exc())
            return f
        return deko

    h_vor = B.sha(Path(B.__file__))
    cfg = copy.deepcopy(CFG)
    rng = np.random.default_rng(11)
    Q = B._q_kunst(260, 0.02, rng, beta_s=0.6, hf_kopplung=0.0)
    Q['dhco3'] = rng.normal(0, 3, len(Q))
    prim, z, _g = B.primaer_kette(Q, cfg)
    q2 = h2_rahmen(prim, cfg)

    @T_('Modelldokumentation: h2_rahmen entspricht den Zeilen der Bindung (h1_modelle plus dropna dS)')
    def _():
        q, _a, _u = B.h1_modelle(prim, cfg)
        assert len(q2) == len(q.dropna(subset=['dS'])) and len(q2) == 260
        a = B.schaetze(q2, 'dS', 'dk', KOV, cfg, seed=SEED_GEBUNDEN, B=300)
        b = B.schaetze(q.dropna(subset=['dS']), 'dS', 'dk', KOV, cfg, seed=cfg['seeds']['bootstrap'] + 1, B=300)
        assert abs(a['beta'] - b['beta']) < 1e-12 and abs(a['ci_lo'] - b['ci_lo']) < 1e-12

    @T_('H2-Schaetzer trifft die Kunststeigung 0,6 innerhalb des Intervalls')
    def _():
        e = B.schaetze(q2, 'dS', 'dk', KOV, cfg, seed=SEED_GEBUNDEN, B=300)
        assert e['ci_lo'] < 0.6 < e['ci_hi'], e

    @T_('Sensitivitaetsprogramm laeuft vollstaendig, Schluessel wie bei H1')
    def _():
        c2 = copy.deepcopy(cfg)
        c2['bootstrap_n'] = 200
        S = sens_h2(Q, prim, q2, Q.copy(), Q.copy(), c2, log=lambda *a: None)
        for k in ('unadjustiert_gleiche_faelle', 'huber', 'einfluss', 'ausgangsbereich', 'nur_anstieg', 'nur_abfall', 'mit_k1', 'mit_bikarbonat', 'nur_abs_dhf_unter_10', 'eine_beobachtung_je_person',
                  'gemischtes_modell', 'ohne_dokumentierte_behandlung', 'alle_erstwerte', 'messfehler_kalium', 'd1_gleiche_paare', 'alle_quellen', 'behandlung_dokumentiert'):
            assert k in S, k
        assert S['huber']['beurteilbar'] and abs(S['huber']['beta'] - 0.6) < 0.3
        assert S['nur_anstieg']['n'] + S['nur_abfall']['n'] == len(q2)

    @T_('Seed-Serie: gleiche Seeds geben gleiche Intervalle, verschiedene Seeds verschiedene')
    def _():
        a = seed_serie(q2, [1, 2, 1], B_n=200, cfg=cfg)
        assert a[0]['ci_lo'] == a[2]['ci_lo'] and a[0]['ci_lo'] != a[1]['ci_lo']
        zs = seed_zusammenfassung(a)
        assert zs['seeds'] == 3 and 0.0 <= zs['anteil_lo_ueber_G'] <= 1.0

    @T_('Seed 255002 mit B=2000 ist ein Praefix von B=4000 (Praefix-Eigenschaft des Bootstraps)')
    def _():
        d = q2
        f = lambda t: np.array([B.steigung(t, 'dS', 'dk', KOV)])
        a = B.cluster_bootstrap(d, 'MRN', f, 300, SEED_GEBUNDEN)
        b = B.cluster_bootstrap(d, 'MRN', f, 600, SEED_GEBUNDEN)
        assert np.allclose(a, b[:300])

    D = B._d_kunst(900, rng, wh_sd=0.02)
    D['steil'] = 5.0 * D.t_amp + rng.normal(0, 0.2, len(D))
    D['steil_rep'] = D.steil + rng.normal(0, 0.25, len(D))
    D['w0_d3'] = np.arange(len(D)) * 1000.0 + 570.0

    @T_('Wiederholbarkeit der Steilheit: SD innerhalb nahe der Kunst-SD, Amplitude wie wh_beschreibung der Bindung')
    def _():
        w = wh_generisch(D, 'steil', 'steil_rep')
        assert abs(w['sw'] - 0.25 / math.sqrt(2)) < 0.02, w
        wa = wh_generisch(D, 't_amp', 't_amp_rep')
        wb = B.wh_beschreibung(D)
        assert wa['n'] == wb['n'] and abs(wa['sw'] - wb['sw']) < 1e-12 and abs(wa['rc'] - wb['rc']) < 1e-12 and abs(wa['icc'] - wb['icc']) < 1e-12
        assert wa['rc_robust'] > 0 and w['icc'] > 0.5
        ends = set(zip(D.CSN[:100], D.t_order[:100]))
        assert wh_generisch(D, 'steil', 'steil_rep', ends)['n'] <= 100

    @T_('Fensterueberlappung wird in der Wiederholbarkeit ausgeschlossen')
    def _():
        d2 = D.copy()
        d2['w0_rep'] = d2['w0_d3'] + 10.0                      # ueberlappt
        w = wh_generisch(d2, 'steil', 'steil_rep')
        assert w.get('n', 0) == 0 or not w.get('beurteilbar', True) or w['n_fensterueberlappung_ausgeschlossen'] == len(D)

    @T_('klinisch_h2: Verhaeltnis zur Wiederholungsgrenze und Stufentabelle')
    def _():
        e = B.schaetze(q2, 'dS', 'dk', KOV, cfg, seed=SEED_GEBUNDEN, B=300)
        w = wh_generisch(D, 'steil', 'steil_rep')
        k = klinisch_h2(e, w, q2)
        assert abs(k['beta_je_rc'] - e['beta'] / w['rc']) < 1e-12 and abs(k['mmol_bis_rc_gruppenmittel'] - w['rc'] / e['beta']) < 1e-12
        assert sum(v['n'] for kk, v in k['individuell_je_kaliumstufe'].items() if 'zusammengefasst' not in kk) == len(q2)
        z = k['individuell_je_kaliumstufe']
        assert z['dk_ab_0.5_aufwaerts_zusammengefasst']['n'] == z['dk_0.5_bis_1']['n'] + z['dk_ab_1_aufwaerts']['n']

    @T_('NK1-Analog: Nullkontrolle ohne Effekt liegt nahe null, Kontrast gerechnet')
    def _():
        Qsh = Q.copy()
        r2 = np.random.default_rng(5)
        Qsh['dS'] = r2.normal(0, 0.3, len(Qsh))
        Qsh['dT'] = r2.normal(0, 0.03, len(Qsh))
        e = nk1_steil(prim, Qsh, cfg, B_n=200)
        assert e['n_paare'] == len(prim) and e['steilheit']['kategorie_nullnaehe'] in ('nahe_null', 'unpraezise')
        assert abs(e['steilheit']['beta']) < 0.3 and 'kontrast_steilheit' in e
        assert abs(e['steilheit']['marge'] - 0.104) < 1e-9

    @T_('NK1-Analog erkennt eine eingebaute Steigung in den versetzten Fenstern')
    def _():
        Qsh = Q.copy()
        Qsh['dS'] = 0.6 * Qsh.dk + np.random.default_rng(6).normal(0, 0.3, len(Qsh))
        e = nk1_steil(prim, Qsh, cfg, B_n=200)
        assert e['steilheit']['kategorie_nullnaehe'] == 'von_null_verschieden'

    @T_('NK2-Analog und Natrium-Analog laufen und erhalten die Nullverteilung')
    def _():
        c2 = copy.deepcopy(cfg)
        c2['nk2_n'] = 80
        e = nk2_steil(prim, c2)
        assert abs(e['null_mittel']) < 0.15 and e['q025'] < e['q975'], e
        n = na_steil(prim, cfg)
        assert 'kategorie_nullnaehe' in n and 'korrelation_dna_dk' in n

    @T_('Skalenvergleich: Standardisierung, Aequivalente und Differenz mit bekannten Kunstwerten')
    def _():
        q = q2.copy()
        s = skalenvergleich(q, 0.05, 0.3, 20.8, cfg, B_n=200)
        bT = B.steigung(q, 'dT', 'dk', KOV)
        bS = B.steigung(q, 'dS', 'dk', KOV)
        assert abs(s['A_zT_je_sd_wiederholung']['wert'] - bT / 0.05) < 1e-12
        assert abs(s['A_zS_je_sd_wiederholung']['wert'] - bS / 0.3) < 1e-12
        assert abs(s['A_differenz_zS_minus_zT']['wert'] - (bS / 0.3 - bT / 0.05)) < 1e-12
        assert abs(s['C_S_als_amplitude_mm_je_mmol_r_spezifikation_20.8']['wert'] - bS / 20.8 * 10) < 1e-12
        assert abs(s['C_differenz_S_minus_r_mal_T_r_spezifikation_20.8']['wert'] - (bS - 20.8 * bT)) < 1e-12
        for k in ('dS_adjustiert_auf_dT', 'dT_adjustiert_auf_dS'):
            assert s[k]['beurteilbar']

    @T_('Skalenvergleich: reine Amplitudenabhaengigkeit (S = r*T) ergibt Differenz S - r*T nahe null und adjustiertes dS ohne Rest')
    def _():
        r2 = np.random.default_rng(21)
        n = 400
        q = B._q_kunst(n, 0.03, r2, sd=0.02)
        q['dS'] = 20.8 * q.dT + r2.normal(0, 0.05, n)
        p, _z, _g = B.primaer_kette(q, cfg)
        qq = h2_rahmen(p, cfg)
        s = skalenvergleich(qq, 0.05, 1.0, 20.8, cfg, B_n=300)
        d = s['C_differenz_S_minus_r_mal_T_r_spezifikation_20.8']
        assert d['ci_lo'] < 0 < d['ci_hi'], d
        a = s['dS_adjustiert_auf_dT']
        assert a['ci_lo'] <= 0 <= a['ci_hi'], a

    @T_('Empirisches Verhaeltnis Steilheit zu Amplitude')
    def _():
        e = empirisches_verhaeltnis(D)
        assert e['beurteilbar'] and 3.0 < e['median_verhaeltnis'] < 8.0

    @T_('Kleinzellenpruefung findet Zaehlungen 1-9, Differenzen und Komplemente; Vergroeberung beseitigt sie')
    def _():
        z = {'gesamt': 216, 'a': 63, 'b': 132, 'rest': 21, 'kind': 210, 'klein': 5}
        v = kleinzellen_pruefung(z, [('kind', 'gesamt'), ('a', 'gesamt')], [('gesamt', ['a', 'b'])])
        typen = {t for t, _l, _n in v}
        assert typen == {'zaehlung', 'differenz'}, v
        z2 = dict(z)
        z2['b'] = 145                                           # 216 - 63 - 145 = 8: Komplement 1..9
        v2 = kleinzellen_pruefung(z2, [], [('gesamt', ['a', 'b'])])
        assert any(t == 'komplement' for t, _l, _n in v2), v2
        g = vergroebern_paarweise(z, [('kind', 'gesamt')], [])
        assert g['klein'][0] == '<10' and g['kind'][1] is True
        cur = {k: v_[0] for k, v_ in g.items() if isinstance(v_[0], int)}
        assert not kleinzellen_pruefung(cur, [('kind', 'gesamt')], [])

    @T_('Gesamtablauf analyse() auf Kunstdaten ohne Zugriff auf MC-MED')
    def _():
        c2 = copy.deepcopy(CFG)
        T = {'Q': Q, 'Qd1': Q.copy(), 'Qa': Q.copy(), 'Qsh': Q.copy(), 'Dm': D}
        T['Qsh']['dS'] = np.random.default_rng(1).normal(0, 0.3, len(Q))
        T['Qsh']['dT'] = np.random.default_rng(2).normal(0, 0.03, len(Q))
        # verkleinerte Ziehungszahlen, damit der Test schnell bleibt
        alt = (CFG['bootstrap_n'], CFG['nk2_n'])
        CFG['bootstrap_n'], CFG['nk2_n'] = 120, 40
        try:
            R = analyse(T, lambda *a: None, ensemble=False, soll=None)
        finally:
            CFG['bootstrap_n'], CFG['nk2_n'] = alt
        for k in ('gate', 'seeds', 'sensitivitaeten', 'stabilitaet_je_split', 'wiederholbarkeit', 'klinisch_h2', 'nk1_steilheit', 'nk2_steilheit', 'natrium_steilheit', 'skalenvergleich'):
            assert k in R, k
        json.dumps(sicher(R))

    @T_('Gebundenes Skript bleibt unveraendert (Hash vor und nach dem Selbsttest)')
    def _():
        assert B.sha(Path(B.__file__)) == h_vor == '04ac15dae0f8a6a0a9b2d154e43b4ff3d240acd410d10707498fa53d48c9dcea'

    @T_('Gebundene Bindung ist intakt (pruefe_start liefert Freigabe, ohne Daten zu lesen)')
    def _():
        fr = B.pruefe_start()
        assert fr.hash

    ok = sum(1 for _n, o, _t in ergebnisse if o)
    print(f'\nSelbsttest: {ok} von {len(ergebnisse)} bestanden')
    return ok == len(ergebnisse)


def lauf(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    log = Protokoll(OUT / 'protokoll.log')
    log('Beginn lauf; Skript', B.sha(Path(__file__))[:16], 'gebundenes Skript', B.sha(Path(B.__file__))[:16])
    with RamReservierung(2.0, 'S3 MC-MED post hoc H2 (1 Prozess)', log):
        soll = json.loads(JSON_GEBUNDEN.read_text(encoding='utf-8'))
        T = tabellen_laden(log, cache=OUT / 'tabellen_cache.pkl')
        try:
            import psutil
            log('RSS nach Laden MB', int(psutil.Process().memory_info().rss / 1e6))
        except Exception:
            pass
        R = analyse(T, log, ensemble='--ohne-ensemble' not in argv, soll=soll)
        try:
            import psutil
            R['rss_mb_ende'] = int(psutil.Process().memory_info().rss / 1e6)
        except Exception:
            pass
    R['skript_sha256'] = B.sha(Path(__file__))
    R['gebundenes_skript_sha256'] = B.sha(Path(B.__file__))
    out = OUT / 'posthoc_h2.json'
    out.write_text(json.dumps(sicher(R), indent=1, ensure_ascii=False), encoding='utf-8')
    log('Ausgabe', out, 'sha256', B.sha(out))
    return R


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
