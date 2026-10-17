"""S3 MC-MED: zweite, kaliumblinde Annotationsseite fuer die NACHTRAEGLICHE PRUEFUNG der einflussreichen Primaerpaare (post hoc, nach Sicht der Ergebnisse).

Anlass (Auftrag K, 17.10.2026): Die H1-Schaetzung (+0,026 mV je mmol/L) haengt an 11 einflussreichen Primaerpaaren (|DFBETAS| der Steigung > 2/sqrt(n) im gebundenen
H1-Primaermodell, gesamt_maskiert.json, h1.sensitivitaeten.einfluss). Ohne sie liegt die Schaetzung nahe null. Die Frage ist, ob die Messkette an diesen Paaren falsch gemessen
hat. Der Anwender (ECG-erfahrener Leser) misst die Streifen von Hand, ohne Kalium, ohne Paarzuordnung und ohne zu wissen, welche Streifen einflussreich sind.
Diese Seite ist NICHT Teil der Bindung und keine Bestaetigung; sie ist beschreibend und post hoc.

Dieses Skript importiert s3_annotation_bauen.py (lies_block, kodiere, baue_html, kunst_streifen, auswahl_hash) und s3_auswertung_v2_2.py UNVERAENDERT. Es ersetzt zur Laufzeit,
im Speicher, einzelne Textstellen der HTML-Vorlage (Titel, Hinweisband, Dateiname des Exports, Zielordner) und stellt die Vorlage danach wieder her. Keine gebundene Datei
wird veraendert.

Modi (Python: <local path> -I):
  selbsttest   Tests an Kunstdaten (kein Zugriff auf MC-MED)
  auswahl      einflussreiche Paare nach der gebundenen Regel bestimmen, Fuellpaare ziehen, Reihenfolge ziehen, Segmente suchen; schreibt AUSWAHL_EINFLUSS.json und
               SCHLUESSEL_EINFLUSS.csv (verweigert, wenn AUSWAHL_EINFLUSS.json schon existiert)
  seite        REFERENZANNOTATION_S3_EINFLUSS.html aus dem Schluessel bauen (prueft Hash und Schluesseldatei)
  kontrolle    gebundene Kette ueber den regulaeren Weg (pruefe_start) an allen gezeigten Enden erneut laufen lassen und gegen die Werte des Gesamtlaufs vergleichen
  testseite    TESTSEITE_EINFLUSS.html mit 3 synthetischen Streifen (keine Daten), optional --ausgabe PFAD

Auswahlregel (vor jeder Messung festgelegt; Seed siehe SEED = 255020, die erste Annotation nutzte 255010):
  * Einflussreiche Paare: genau die Regel von s3_auswertung_v2_2.einfluss_pruefung auf dem H1-Primaermodell (Zeilen von h1_modelle(prim): dropna auf dT, dk und die HF-Spline-
    spalten; OLS dT ~ dk + dB1..dBk mit freiem Achsenabschnitt; |DFBETAS| der Steigung > 2/sqrt(n)). Gate: Zahl, groesster Wert und Grenze gleich den gebundenen Zahlen.
  * Beide Enden jedes einflussreichen Paars. Streifen = 10 s Ableitung II, zentriert auf die Mitte des 60-s-Messfensters, das die Kette fuer dieses Ende verwendet hat
    ((w0 + w1) / 2 aus dem gebundenen Gesamtlauf; im Regelfall gleich der Blockmitte).
  * Fuellpaare: ebenso viele nicht einflussreiche Primaerpaare, zufaellig gezogen, nur Train und Val (der Testsplit wird nur fuer die einflussreichen Paare gelesen, die darin
    liegen), je Person hoechstens ein Paar, keine Person eines einflussreichen Paars, beide Enden ohne zeitliche Ueberschneidung mit einem Streifen der ersten Annotation.
  * Einflussreiche Paare bleiben auch bei Ueberschneidung mit der ersten Annotation (Vermerk im Schluessel).
  * Reihenfolge zufaellig; die beiden Enden eines Paars liegen mindestens MIN_ABSTAND_ENDEN Ansichten auseinander. Ein Streifen, der Ende mehrerer Paare ist, wird nur einmal gezeigt.
  * Die Kennzeichnung einflussreich ja/nein steht nur in SCHLUESSEL_EINFLUSS.csv.
Regeln: keine Kennungen in der Seite; Einzelprozess unter 1,5 GB; keine Loeschbefehle; Ausgaben mit Daten nur unter <local path>.
Jeder datenlesende Modus verlangt, dass der SHA-256 dieses Skripts in koordination\\TESTZUGRIFFE.md eingetragen ist (der Testsplit liegt in der Primaermenge).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, os, json, time, math, hashlib, argparse, contextlib, re, pickle, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import s3_annotation_bauen as AB            # gebunden, unveraendert, nur importiert

ANN1 = Path(str(data_path("s3.annotation_dir")))                       # erste Annotation (nur lesen)
ANN = Path(str(data_path("s3.annotation_influence_dir")))
CACHE = Path(str(data_path("s3.posthoc_h2_table_cache_pkl")))   # Tabellen des gebundenen Gesamtlaufs (enthaelt den Testsplit-Anteil)
GEBUNDEN_JSON = Path(str(data_path("s3.masked_result_json")))
LOCK = Path(str(data_path("s3.spec_lock_json")))
TESTZUGRIFFE = Path(str(data_path("s3.test_access_register_md")))
SEED = 255020                      # fester Seed dieser Ziehung (Fuellpaare: [SEED, 1], Reihenfolge: [SEED, 2])
MIN_ABSTAND_ENDEN = 8              # Ansichten zwischen den beiden Enden eines Paars (mindestens)
FUELL_FAKTOR = 1.0                 # Fuellpaare je einflussreiches Paar
MIN_FUELL_STREIFEN = 14            # Untergrenze der Fuellstreifen, falls die Messzeit gekuerzt werden muss (hier nicht gebraucht)
UEBERSCHNEIDUNG_S = 10.0           # zwei Streifen desselben Besuchs mit Mittenabstand < 10 s ueberschneiden sich
VERSION = 'annot-einfluss-v1'
FS = 500


# =========================================================================== Hilfen
def sha_datei(p):
    return AB.sha_datei(p)


def log(*a):
    s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    if ANN.exists():
        with open(ANN / 'protokoll_einfluss.log', 'a', encoding='utf-8') as f:
            f.write(s + '\n')


def pruefe_eintrag():
    """Jeder datenlesende Modus verlangt den Eintrag des Skript-Hashes in TESTZUGRIFFE.md (Testsplit liegt in der Primaermenge)."""
    h = sha_datei(Path(__file__))
    txt = TESTZUGRIFFE.read_text(encoding='utf-8')
    if h not in txt:
        raise SystemExit(f'Skript-Hash {h} fehlt in {TESTZUGRIFFE}; zuerst den Eintrag anhaengen')
    return h


# =========================================================================== Einflussregel (gebunden) und Primaerrahmen
def dfbetas_regel(d, kov, y='dT', x='dk'):
    """Rechenvorschrift von s3_auswertung_v2_2.einfluss_pruefung, Zeile fuer Zeile: OLS mit freiem Achsenabschnitt, |DFBETAS| der Steigung von x, Grenze 2 / sqrt(n).
    d muss die Zeilen des Modells enthalten (bereits ohne fehlende Werte). Rueckgabe (dfb, grenze)."""
    from statsmodels.stats.outliers_influence import OLSInfluence
    import statsmodels.api as sm
    d = d.reset_index(drop=True)
    X = np.column_stack([np.ones(len(d)), d[x].to_numpy()] + [d[k].to_numpy() for k in kov])
    m = sm.OLS(d[y].to_numpy(), X).fit()
    dfb = np.abs(OLSInfluence(m).dfbetas[:, 1])
    return dfb, 2.0 / math.sqrt(len(d))


def primaerrahmen(Q, B):
    """Die Zeilen des gebundenen H1-Modells (h1_modelle): Primaermenge, HF-Spline, dropna auf dT, dk und die Splinespalten; Index 0..n-1 wie in einfluss_pruefung."""
    prim, z, _g = B.primaer_kette(Q, B.CFG)
    kov = B.hf_spalten(B.CFG)
    q = B.mit_hf_spline(prim, B.CFG).dropna(subset=['dT', 'dk'] + kov).reset_index(drop=True)
    return q, kov, z


# =========================================================================== Ziehung (reine Funktionen, ohne Daten testbar)
def mitte_fenster(d, ende):
    """Mitte des 60-s-Messfensters der Kette an Ende 'a' oder 'b'."""
    return ((d['w0_' + ende] + d['w1_' + ende]) / 2.0).to_numpy(float)


def runde1_karte(r1):
    """r1: DataFrame mit CSN, t_mitte (erste Annotation) -> {CSN: ndarray der Mitten}."""
    return {int(c): g.t_mitte.to_numpy(float) for c, g in r1.groupby('CSN')}


def ueberschneidet(karte, csn, mid, grenze=UEBERSCHNEIDUNG_S):
    a = karte.get(int(csn))
    return bool(a is not None and np.any(np.abs(a - mid) < grenze))


def waehle(d, dfb, grenze, r1, seed=SEED, fuell_faktor=FUELL_FAKTOR, min_abstand=MIN_ABSTAND_ENDEN):
    """Reine Auswahl. d: Primaerrahmen (Zeilen 0..n-1, Spalten CSN, MRN, split, ta, tb, w0_a, w1_a, w0_b, w1_b, ...), dfb/grenze: DFBETAS-Regel, r1: erste Annotation (CSN, t_mitte).
    Rueckgabe dict: paare (DataFrame je Paar), enden (long, eine Zeile je Paar und Ende, mit ansicht_nr), streifen (eine Zeile je gezeigtem Streifen), info."""
    d = d.reset_index(drop=True)
    n = len(d)
    infl = [int(i) for i in np.where(np.asarray(dfb) > grenze)[0]]
    infl_set = set(infl)
    karte = runde1_karte(r1)
    mid = {e: mitte_fenster(d, e) for e in ('a', 'b')}
    ueb = np.array([[ueberschneidet(karte, d.CSN[i], mid[e][i]) for e in ('a', 'b')] for i in range(n)])
    besuch_r1 = np.array([int(d.CSN[i]) in karte for i in range(n)])
    # Fuellpaare
    kand = [i for i in range(n) if i not in infl_set and d.split[i] in ('train', 'val') and not ueb[i].any()]
    n_fuell = int(round(fuell_faktor * len(infl)))
    rng_f = np.random.default_rng([seed, 1])
    perm = rng_f.permutation(len(kand))
    benutzt = {d.MRN[i] for i in infl}
    fuell = []
    for k in perm:
        i = kand[int(k)]
        if d.MRN[i] in benutzt:
            continue
        fuell.append(i)
        benutzt.add(d.MRN[i])
        if len(fuell) == n_fuell:
            break
    if len(fuell) < n_fuell:
        raise ValueError(f'zu wenige Fuellpaare: {len(fuell)} von {n_fuell}')
    # Paare und Enden
    reihe = [(i, True) for i in sorted(infl, key=lambda i: (int(d.CSN[i]), float(d.ta[i])))] + [(i, False) for i in sorted(fuell, key=lambda i: (int(d.CSN[i]), float(d.ta[i])))]
    paare, enden = [], []
    for k, (i, ist_infl) in enumerate(reihe):
        pid = f'P{k + 1:02d}'
        paare.append({'paar_id': pid, 'zeile': i, 'einflussreich': 'ja' if ist_infl else 'nein', 'CSN': int(d.CSN[i]), 'MRN': int(d.MRN[i]), 'split': d.split[i],
                      'dfbetas': float(dfb[i]), 'ueb_a': bool(ueb[i, 0]), 'ueb_b': bool(ueb[i, 1])})
        for j, e in enumerate(('a', 'b')):
            enden.append({'paar_id': pid, 'ende': e, 'zeile': i, 'einflussreich': 'ja' if ist_infl else 'nein', 'CSN': int(d.CSN[i]), 'MRN': int(d.MRN[i]), 'split': d.split[i],
                          't_mitte': round(float(mid[e][i]), 3), 'ueberschneidung_runde1': 'ja' if ueb[i, j] else 'nein', 'besuch_in_runde1': 'ja' if besuch_r1[i] else 'nein'})
    enden = pd.DataFrame(enden)
    paare = pd.DataFrame(paare)
    # Streifen (ein Streifen = ein Besuch und eine Mitte; ein geteiltes Ende wird nur einmal gezeigt)
    enden['streifen_id'] = enden.CSN.astype(str) + '|' + enden.t_mitte.map(lambda v: f'{v:.3f}')
    uniq = list(dict.fromkeys(enden.streifen_id))
    idx = {s: k for k, s in enumerate(uniq)}
    paar_idx = [(idx[g.streifen_id.iloc[0]], idx[g.streifen_id.iloc[1]]) for _, g in enden.groupby('paar_id', sort=True)]
    perm_o = ordne(len(uniq), [p for p in paar_idx if p[0] != p[1]], seed, min_abstand)
    nr = np.empty(len(uniq), int)
    nr[perm_o] = np.arange(1, len(uniq) + 1)
    enden['ansicht_nr'] = [int(nr[idx[s]]) for s in enden.streifen_id]
    streifen = enden.drop_duplicates('streifen_id')[['ansicht_nr', 'streifen_id', 'CSN', 't_mitte']].sort_values('ansicht_nr').reset_index(drop=True)
    info = {'n_primaer': int(n), 'n_einflussreich': len(infl), 'n_fuellpaare': len(fuell), 'fuell_kandidaten': len(kand), 'fuell_ausgeschlossen_ueberschneidung_runde1':
            int(sum(1 for i in range(n) if i not in infl_set and d.split[i] in ('train', 'val') and ueb[i].any())), 'fuell_ausgeschlossen_testsplit': int(sum(1 for i in range(n) if i not in infl_set and d.split[i] == 'test')),
            'einflussreich_mit_ueberschneidung_runde1_enden': int(ueb[infl].sum()) if infl else 0, 'einflussreich_split': {s: int(v) for s, v in d.split[infl].value_counts().items()},
            'fuell_split': {s: int(v) for s, v in d.split[fuell].value_counts().items()}, 'streifen': int(len(uniq)), 'enden': int(len(enden)),
            'geteilte_streifen': int(len(enden) - len(uniq))}
    return {'paare': paare, 'enden': enden, 'streifen': streifen, 'info': info}


def ordne(n_items, paare, seed=SEED, min_abstand=MIN_ABSTAND_ENDEN, versuche=3000000):
    """Zufaellige Reihenfolge der Streifen (Rueckgabe: perm, perm[k] = Streifen an Position k), bei der die beiden Enden jedes Paars mindestens min_abstand Positionen
    auseinander liegen (Zurueckweisungsverfahren, fester Zufallsstrom [seed, 2])."""
    rng = np.random.default_rng([seed, 2])
    pa = np.array(paare, int).reshape(-1, 2)
    pos = np.empty(n_items, int)
    for _ in range(versuche):
        perm = rng.permutation(n_items)
        pos[perm] = np.arange(n_items)
        if len(pa) == 0 or bool(np.all(np.abs(pos[pa[:, 0]] - pos[pa[:, 1]]) >= min_abstand)):
            return perm
    raise ValueError('keine Reihenfolge mit Mindestabstand gefunden')


# =========================================================================== Segmente (Daten: nur Koepfe)
def finde_segment(B, csn, w0, w1):
    """Ableitung-II-Segment (mV, 500 Hz), das das ganze Messfenster [w0, w1] enthaelt: (Segmentnr, n, t0, t1). Genau ein Treffer, sonst Fehler."""
    p = B.WAVE / str(csn)[-3:] / str(csn) / 'II'
    treffer = []
    for f in sorted(os.listdir(p)):
        if not f.endswith('.hea'):
            continue
        ns, fs, n, t0, einheit = B.lies_kopf(p / f)
        if str(einheit).lower() != 'mv' or fs != FS:
            continue
        t1 = t0 + n / fs
        if t0 - 1e-3 <= w0 and w1 <= t1 + 1e-3:
            treffer.append((int(f[:-4].rsplit('_', 1)[1]), int(n), float(t0), float(t1)))
    if len(treffer) != 1:
        raise ValueError(f'{len(treffer)} Segmente enthalten das Messfenster (erwartet 1)')
    return treffer[0]


# =========================================================================== Schluessel, Auswahl-Datei
SPALTEN = ['ansicht_nr', 'paar_id', 'ende', 'einflussreich', 'CSN', 'MRN', 'split', 'ta', 'tb', 'seg', 't_mitte', 'w0', 'w1', 'c_kette', 'pol_kette', 't_amp_kette',
           't_roh_kette', 'qrs_ms_kette', 'hf', 'ka', 'kb', 'dk', 'dT_kette', 'dfbetas', 'ueberschneidung_runde1', 'besuch_in_runde1']


def schluessel_bauen(sel, d, B):
    """Langes Format (eine Zeile je Paar und Ende) mit Segmentnummer und den Werten der gebundenen Kette; Segment ueber die Kopfdateien."""
    zeilen = []
    segcache = {}
    for r in sel['enden'].itertuples():
        i = int(r.zeile)
        e = r.ende
        w0, w1 = float(d['w0_' + e][i]), float(d['w1_' + e][i])
        key = (int(r.CSN), round(w0, 3))
        if key not in segcache:
            segcache[key] = finde_segment(B, int(r.CSN), w0, w1)
        sn, n, t0, t1 = segcache[key]
        assert t0 <= r.t_mitte - 5.0 + 1e-3 and r.t_mitte + 5.0 <= t1 + 1e-3, 'Streifen ragt aus dem Segment'
        pr = sel['paare'].set_index('paar_id').loc[r.paar_id]
        zeilen.append({'ansicht_nr': int(r.ansicht_nr), 'paar_id': r.paar_id, 'ende': e, 'einflussreich': r.einflussreich, 'CSN': int(r.CSN), 'MRN': int(r.MRN), 'split': r.split,
                       'ta': round(float(d.ta[i]), 3), 'tb': round(float(d.tb[i]), 3), 'seg': sn, 't_mitte': r.t_mitte, 'w0': round(w0, 3), 'w1': round(w1, 3),
                       'c_kette': round(float(d['c_' + e][i]), 3), 'pol_kette': int(d['pol_' + e][i]), 't_amp_kette': float(d['t_amp_' + e][i]), 't_roh_kette': float(d['t_roh_' + e][i]),
                       'qrs_ms_kette': float(d['qrs_ms_' + e][i]), 'hf': float(d['hf_' + e][i]), 'ka': float(d.ka[i]), 'kb': float(d.kb[i]), 'dk': float(d.dk[i]),
                       'dT_kette': float(d.dT[i]), 'dfbetas': float(pr.dfbetas), 'ueberschneidung_runde1': r.ueberschneidung_runde1, 'besuch_in_runde1': r.besuch_in_runde1})
    K = pd.DataFrame(zeilen, columns=SPALTEN).sort_values(['ansicht_nr', 'paar_id', 'ende']).reset_index(drop=True)
    return K


def streifen_aus_schluessel(K):
    """Eine Zeile je gezeigtem Streifen (ansicht_nr, CSN, seg, c = Streifenmitte) fuer Hash und Seitenbau."""
    s = K.drop_duplicates('ansicht_nr')[['ansicht_nr', 'CSN', 'seg', 't_mitte']].sort_values('ansicht_nr').reset_index(drop=True)
    s = s.rename(columns={'t_mitte': 'c'})
    s['wiederholung_von'] = 0
    return s


# =========================================================================== Seite
def vorlage_einfluss(vorlage=None):
    """HTML-Vorlage der ersten Annotation mit vier Textersetzungen (im Speicher): Titel, Hinweisband, Exportdateiname, Zielordner. Jede Ersetzung muss ihre Stelle finden."""
    h = vorlage if vorlage is not None else AB.HTML_VORLAGE
    band = ('<div class="banner warn" id="postbanner">Nachträgliche Prüfung (post hoc). Diese Seite ist von der ersten Referenzannotation getrennt: eigener Zwischenstand im Browser, '
            'eigene Exportdatei. Bitte genau so messen wie dort. Es gibt keine Angaben zu den Streifen und keinen Hinweis, welche Streifen zusammengehören.</div>')
    ers = [('<title>Referenzannotation T-Amplitude</title>', '<title>Referenzannotation T-Amplitude, nachträgliche Prüfung</title>', 1),
           ('<h1 id="titel">Referenzannotation T-Amplitude</h1>', '<h1 id="titel">Referenzannotation T-Amplitude (nachträgliche Prüfung, post hoc)</h1>', 1),
           ('<div id="banners"></div>', '<div id="banners">' + band + '</div>', 1),
           ("'referenzannotation_s3_ergebnisse_'", "'referenzannotation_s3_einfluss_'", 1),
           ('s3\\\\annotation\\\\', 's3\\\\annotation_einfluss\\\\', 2)]
    for alt, neu, anz in ers:
        if h.count(alt) != anz:
            raise AssertionError(f'Vorlage: Stelle {alt!r} {h.count(alt)}-mal gefunden statt {anz}-mal')
        h = h.replace(alt, neu)
    return h


@contextlib.contextmanager
def _vorlage_patch():
    alt_v, alt_ver = AB.HTML_VORLAGE, AB.VERSION
    neu = vorlage_einfluss(alt_v)
    AB.HTML_VORLAGE, AB.VERSION = neu, VERSION
    try:
        yield
    finally:
        AB.HTML_VORLAGE, AB.VERSION = alt_v, alt_ver


def baue_seite(ansichten, hash_, testmodus=False):
    """Seite mit der veraenderten Vorlage ueber AB.baue_html. Eigener localStorage-Schluessel."""
    key = 's3einfl_' + ('testseite' if testmodus else hash_[:16]) + '_' + VERSION
    with _vorlage_patch():
        h = AB.baue_html(ansichten, hash_, key, testmodus=testmodus)
    assert AB.HTML_VORLAGE.count('referenzannotation_s3_ergebnisse_') == 1, 'Vorlage nicht wiederhergestellt'
    return h


def modus_testseite(ziel=None):
    faelle = [(0.30, 1.20, 0.05), (-0.20, 1.00, -0.10), (0.50, -1.10, 0.00)]
    ans = []
    for k, (ta, qa, off) in enumerate(faelle):
        x, rs = AB.kunst_streifen(ta, qa, off, hr=60.0, seed=11 + k)
        dd = AB.kodiere(x)
        dd['nr'] = k + 1
        dd['test'] = {'t_amp_mv': ta, 'qrs_mv': qa, 'offset_mv': off, 'r_s': [float(r) for r in rs], 'spitze_nach_r_s': 0.28, 'basis_nach_r_s': 0.50}
        ans.append(dd)
    h = baue_seite(ans, AB.sha_text('testseite_einfluss'), testmodus=True)
    ziel = Path(ziel) if ziel else ANN / 'TESTSEITE_EINFLUSS.html'
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(h, encoding='utf-8')
    print('geschrieben', ziel, f'{len(h) / 1e6:.2f} MB')


# =========================================================================== Modi mit Daten
def lade_tabellen():
    with open(CACHE, 'rb') as f:
        return pickle.load(f)


def modus_auswahl():
    sha_self = pruefe_eintrag()
    ANN.mkdir(parents=True, exist_ok=True)
    pa, ps = ANN / 'AUSWAHL_EINFLUSS.json', ANN / 'SCHLUESSEL_EINFLUSS.csv'
    if pa.exists() or ps.exists():
        raise SystemExit('AUSWAHL_EINFLUSS.json oder SCHLUESSEL_EINFLUSS.csv existiert bereits; die Auswahl wird nicht neu gezogen')
    import s3_auswertung_v2_2 as B
    log('Beginn auswahl; Skript', sha_self[:16], '; gebundenes Skript', sha_datei(Path(B.__file__))[:16])
    T = lade_tabellen()
    q, kov, zk = primaerrahmen(T['Q'], B)
    log('Primaerrahmen H1:', len(q), 'Zeilen; Split', q.split.value_counts().to_dict())
    soll = json.loads(GEBUNDEN_JSON.read_text(encoding='utf-8'))
    sh = soll['h1']['sensitivitaeten']['einfluss']
    sp = soll['h1']['primaer']
    # Gate 1: H1-Punktschaetzer (Seed 255001, B = 2000) wie gebunden
    adj = B.schaetze(q, 'dT', 'dk', kov, B.CFG)
    assert adj['n'] == sp['n'] and abs(adj['beta'] - sp['beta']) < 1e-12 and abs(adj['ci_lo'] - sp['ci_lo']) < 1e-12 and abs(adj['ci_hi'] - sp['ci_hi']) < 1e-12, 'H1 weicht von der Bindung ab'
    log('Gate H1 bestanden: n', adj['n'], 'beta', adj['beta'])
    # Gate 2: Einflussregel (Eigenrechnung gegen die gebundene Funktion und gegen die gebundene JSON)
    dfb, grenze = dfbetas_regel(q, kov)
    ref = B.einfluss_pruefung(q, 'dT', 'dk', kov, B.CFG)
    n_infl = int((dfb > grenze).sum())
    assert n_infl == ref['n_einflussreich'] == sh['n_einflussreich'], (n_infl, ref['n_einflussreich'], sh['n_einflussreich'])
    assert abs(float(dfb.max()) - ref['dfbetas_max']) < 1e-12 and abs(float(dfb.max()) - sh['dfbetas_max']) < 1e-12 and abs(grenze - sh['grenze']) < 1e-12
    assert abs(ref['ohne_einflussreiche']['beta'] - sh['ohne_einflussreiche']['beta']) < 1e-12
    log('Gate Einflussregel bestanden:', n_infl, 'einflussreiche Paare, groesster |DFBETAS|', float(dfb.max()), ', Grenze', grenze, ', Schaetzer ohne sie', ref['ohne_einflussreiche']['beta'])
    r1 = pd.read_csv(ANN1 / 'SCHLUESSEL.csv')[['CSN', 't_mitte']]
    sel = waehle(q, dfb, grenze, r1)
    inf = sel['info']
    log('Auswahl:', json.dumps(inf))
    K = schluessel_bauen(sel, q, B)
    # Pruefungen
    assert K.groupby('ansicht_nr').CSN.nunique().max() == 1
    assert (K.groupby('paar_id').ansicht_nr.nunique() == 2).all(), 'ein Paar zeigt beide Enden als denselben Streifen'
    assert set(K[K.einflussreich == 'nein'].split) <= {'train', 'val'}
    ps_ = streifen_aus_schluessel(K)
    ah = AB.auswahl_hash(ps_)
    ANN.mkdir(parents=True, exist_ok=True)
    K.to_csv(ps, index=False)
    ab = K.sort_values('ansicht_nr')
    pos = {p: sorted(g.ansicht_nr) for p, g in ab.groupby('paar_id')}
    abst = [v[1] - v[0] for v in pos.values()]
    info = {
        'art': 'Auswahl der Streifen fuer die zweite, kaliumblinde Annotationsseite (nachtraegliche Pruefung der einflussreichen H1-Primaerpaare, S3 MC-MED)',
        'kennzeichnung': 'NACHTRAEGLICHE PRUEFUNG, POST HOC, nach Sicht der Hauptergebnisse veranlasst; beschreibend, nicht gebunden, keine Bestaetigung',
        'erstellt': dt.datetime.now().astimezone().isoformat(timespec='seconds'),
        'version': VERSION, 'seed': SEED, 'seed_stroeme': {'fuellpaare': [SEED, 1], 'reihenfolge': [SEED, 2]},
        'regel': [
            'Einflussreiche Paare: genau die Regel von s3_auswertung_v2_2.einfluss_pruefung auf dem H1-Primaermodell (Zeilen von h1_modelle: dropna dT, dk, HF-Splinespalten; OLS dT ~ dk + dB '
            'mit freiem Achsenabschnitt; |DFBETAS| der Steigung > 2/sqrt(n)). Gate: Zahl, groesster Wert, Grenze und Schaetzer ohne sie gleich gesamt_maskiert.json.',
            'Beide Enden jedes einflussreichen Paars; Streifen = 10 s Ableitung II, zentriert auf die Mitte des 60-s-Messfensters der Kette fuer dieses Ende ((w0 + w1) / 2).',
            f'Fuellpaare: {FUELL_FAKTOR:g} Fuellpaar je einflussreiches Paar, zufaellig aus den nicht einflussreichen Primaerpaaren, nur Train und Val, je Person hoechstens ein Paar, keine Person '
            'eines einflussreichen Paars, beide Enden ohne Ueberschneidung (gleicher Besuch und Mittenabstand < 10 s) mit einem Streifen der ersten Annotation.',
            'Einflussreiche Paare bleiben bei Ueberschneidung mit der ersten Annotation (Vermerk im Schluessel).',
            f'Reihenfolge zufaellig; die beiden Enden eines Paars liegen mindestens {MIN_ABSTAND_ENDEN} Ansichten auseinander; gleicher Streifen als Ende mehrerer Paare wird nur einmal gezeigt.',
            'Die Kennzeichnung einflussreich ja/nein steht nur in SCHLUESSEL_EINFLUSS.csv.'],
        'einflussregel': {'n_primaer': int(len(q)), 'n_einflussreich': n_infl, 'grenze': grenze, 'dfbetas_max': float(dfb.max()), 'h1_beta_alle': adj['beta'], 'h1_beta_ohne_einflussreiche': ref['ohne_einflussreiche']['beta']},
        'ziehung': inf,
        'ansichten': int(len(ps_)), 'enden_paare_abstand_min': int(min(abst)), 'enden_paare_abstand_median': float(np.median(abst)),
        'testsplit': ('Der Testsplit wird nur fuer die einflussreichen Paare gelesen, die darin liegen (Eintrag in koordination/TESTZUGRIFFE.md); Fuellpaare nur Train und Val. Die '
                      'Primaertabellen kommen aus dem Zwischenspeicher des gebundenen Gesamtlaufs (tabellen_cache.pkl).'),
        'auswahl_hash': ah,
        'auswahl_hash_bildung': 'SHA-256 ueber die Zeilen "ansicht_nr|CSN|Streifenmitte (3 Nachkommastellen)|0", durch Zeilenumbruch getrennt, in Ansichtsreihenfolge',
        'schluessel_datei_sha256': sha_datei(ps),
        'software_sha256': {p.name: sha_datei(p) for p in [Path(__file__), HIER / 's3_annotation_bauen.py', HIER / 's3_auswertung_v2_2.py']},
        'hinweis_software': 'Die Ersetzungen der Vorlage geschehen im Speicher; s3_annotation_bauen.py wurde nicht veraendert.',
    }
    pa.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')
    log('Auswahl geschrieben; Ansichten', len(ps_), '; Hash', ah)
    print('auswahl_hash', ah)


def _lies_schluessel():
    pa, ps = ANN / 'AUSWAHL_EINFLUSS.json', ANN / 'SCHLUESSEL_EINFLUSS.csv'
    if not pa.exists() or not ps.exists():
        raise SystemExit('AUSWAHL_EINFLUSS.json oder SCHLUESSEL_EINFLUSS.csv fehlt; zuerst Modus auswahl')
    info = json.loads(pa.read_text(encoding='utf-8'))
    if sha_datei(ps) != info['schluessel_datei_sha256']:
        raise SystemExit('SCHLUESSEL_EINFLUSS.csv weicht von AUSWAHL_EINFLUSS.json ab')
    K = pd.read_csv(ps)
    if AB.auswahl_hash(streifen_aus_schluessel(K)) != info['auswahl_hash']:
        raise SystemExit('Auswahl-Hash stimmt nicht')
    return info, K


def modus_seite():
    pruefe_eintrag()
    info, K = _lies_schluessel()
    s = streifen_aus_schluessel(K)
    ans = []
    for r in s.itertuples():
        x = AB.lies_block(int(r.CSN), int(r.seg), float(r.c))
        d = AB.kodiere(x)
        ans.append({'nr': int(r.ansicht_nr), **d})
    h = baue_seite(ans, info['auswahl_hash'])
    ohne_daten = re.sub(r'"b64":"[^"]*"', '"b64":""', h)
    for v in list(K.CSN.astype(str).unique()) + list(K.MRN.astype(str).unique()):
        assert v not in ohne_daten, 'Kennung in der Seite'
    assert 'einflussreich' not in ohne_daten.lower().replace('nachträgliche prüfung', '')
    ziel = ANN / 'REFERENZANNOTATION_S3_EINFLUSS.html'
    ziel.write_text(h, encoding='utf-8')
    log('Seite geschrieben', ziel, f'{len(h) / 1e6:.2f} MB', 'Ansichten', len(ans), 'sha256', sha_datei(ziel))
    print('geschrieben', ziel, f'{len(h) / 1e6:.2f} MB', 'Ansichten', len(ans), 'Hash', info['auswahl_hash'])


def modus_kontrolle():
    """Die gebundene Kette laeuft ueber den regulaeren Weg (pruefe_start, setze_bindung, messe_bei) an jedem gezeigten Ende (Blockmitte c der Kette) und wird mit den Werten
    des Gesamtlaufs im Schluessel verglichen. Prueft auch, dass die Streifenmitte die Mitte des Messfensters ist und der Streifen ganz im Fenster liegt."""
    pruefe_eintrag()
    info, K = _lies_schluessel()
    import s3_auswertung_v2_2 as V
    fr = V.pruefe_start(LOCK)
    V.setze_bindung(fr)
    log('Bindung geprueft (pruefe_start): Pipeline', V.PIPELINE, '; Freigabe-Hash', fr.hash[:16])
    zeilen = []
    for r in K.itertuples():
        p = V.WAVE / str(r.CSN)[-3:] / str(r.CSN) / 'II' / f'{r.CSN}_{r.seg}'
        ns, fs, n, t0, einheit = V.lies_kopf(str(p) + '.hea')
        seg = (int(r.seg), float(fs), int(n), float(t0), float(t0 + n / fs))
        m = V.messe_bei(int(r.CSN), seg, float(r.c_kette), V.CFG, V.WAVE)
        z = {'ansicht_nr': int(r.ansicht_nr), 'gemessen': m is not None}
        if m is not None:
            z.update(valid=int(m['valid']), d_t_amp=float(abs(m['t_amp'] - r.t_amp_kette)) if np.isfinite(m['t_amp']) else float('nan'), d_pol=int(m['pol'] != r.pol_kette),
                     d_w0=float(abs(m['w0'] - r.w0)), d_w1=float(abs(m['w1'] - r.w1)),
                     strip_im_fenster=bool(m['w0'] - 1e-3 <= r.t_mitte - 5.0 and r.t_mitte + 5.0 <= m['w1'] + 1e-3),
                     strip_mitte_gleich_fenstermitte=bool(abs((m['w0'] + m['w1']) / 2 - r.t_mitte) < 2e-3))
        zeilen.append(z)
    Z = pd.DataFrame(zeilen)
    ok = Z[Z.gemessen]
    res = {'enden': int(len(Z)), 'gemessen': int(Z.gemessen.sum()), 'valid': int(ok.valid.sum()), 'max_abw_t_amp': float(ok.d_t_amp.max()), 'abweichende_pol': int(ok.d_pol.sum()),
           'max_abw_fenster_s': float(max(ok.d_w0.max(), ok.d_w1.max())), 'strip_im_fenster': int(ok.strip_im_fenster.sum()), 'strip_mitte_gleich_fenstermitte': int(ok.strip_mitte_gleich_fenstermitte.sum()),
           'erstellt': dt.datetime.now().astimezone().isoformat(timespec='seconds'), 'freigabe_hash': fr.hash, 'skript_sha256': sha_datei(Path(__file__))}
    (ANN / 'KONTROLLE_KETTE.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    log('Kontrolle:', json.dumps(res))


# =========================================================================== Selbsttest (Kunstdaten)
def kunst_rahmen(n=216, seed=3, n_infl=11):
    """Kuenstlicher Primaerrahmen: dT = beta * dk + Rauschen, n_infl Hebelpunkte mit grossem Einfluss."""
    rng = np.random.default_rng(seed)
    dk = rng.normal(0, 0.9, n)
    kov = ['dB1', 'dB2']
    d = pd.DataFrame({'CSN': np.arange(n) + 9000, 'MRN': np.repeat(np.arange(n // 2) + 100, 2)[:n], 'split': rng.choice(['train', 'val', 'test'], n, p=[0.8, 0.1, 0.1]),
                      'ta': rng.uniform(1e9, 2e9, n), 'dk': dk, 'dB1': rng.normal(0, 1, n), 'dB2': rng.normal(0, 1, n)})
    d['tb'] = d.ta + 7200
    d['dT'] = 0.0 * d.dk + rng.normal(0, 0.15, n)
    hebel = rng.choice(n, n_infl, replace=False)
    d.loc[hebel, 'dk'] = rng.choice([-1, 1], n_infl) * 2.0
    d.loc[hebel, 'dT'] = d.loc[hebel, 'dk'] * 0.4
    for e, off in (('a', 0.0), ('b', 3600.0)):
        c = d.ta + off
        d['c_' + e], d['w0_' + e], d['w1_' + e] = c, c - 30, c + 30
        d['pol_' + e], d['t_amp_' + e], d['t_roh_' + e], d['qrs_ms_' + e], d['hf_' + e] = 1, 0.3, 0.3, 90.0, 70.0
    d['ka'], d['kb'] = 4.0, 4.0 + d.dk
    return d, kov


def selbsttest():
    ok = 0
    # 1. Einflussregel findet geplante Hebelpunkte
    d, kov = kunst_rahmen()
    dfb, gr = dfbetas_regel(d, kov)
    infl = np.where(dfb > gr)[0]
    assert abs(gr - 2 / math.sqrt(216)) < 1e-15 and len(infl) >= 5
    ok += 1
    # 2. Auswahl: Regeln
    rng = np.random.default_rng(8)
    r1 = pd.DataFrame({'CSN': d.CSN.sample(40, random_state=1).to_numpy(), 't_mitte': 0.0})
    r1['t_mitte'] = [float(d.loc[d.CSN == c, 'ta'].iloc[0]) + 3.0 for c in r1.CSN]       # 3 s neben der Mitte des Enden a -> ueberschneidet
    s = waehle(d, dfb, gr, r1)
    inf = s['info']
    assert inf['n_einflussreich'] == len(infl) and inf['n_fuellpaare'] == len(infl)
    P, En = s['paare'], s['enden']
    fu = P[P.einflussreich == 'nein']
    assert set(fu.split) <= {'train', 'val'} and fu.MRN.is_unique and not (set(fu.MRN) & set(P[P.einflussreich == 'ja'].MRN))
    assert not ((En.einflussreich == 'nein') & (En.ueberschneidung_runde1 == 'ja')).any(), 'Fuellpaar mit Ueberschneidung'
    assert set(P[P.einflussreich == 'ja'].zeile) == set(int(i) for i in infl)
    assert sorted(s['streifen'].ansicht_nr) == list(range(1, len(s['streifen']) + 1))
    ok += 1
    # 3. Einflussreiche bleiben auch bei Ueberschneidung; Vermerk gesetzt
    r1b = pd.DataFrame({'CSN': [int(d.CSN[infl[0]])], 't_mitte': [float(d.ta[infl[0]]) + 1.0]})
    s2 = waehle(d, dfb, gr, r1b)
    e0 = s2['enden'][(s2['enden'].zeile == int(infl[0]))]
    assert (e0.einflussreich == 'ja').all() and (e0.ueberschneidung_runde1 == 'ja').sum() == 1 and s2['info']['einflussreich_mit_ueberschneidung_runde1_enden'] == 1
    ok += 1
    # 4. Determinismus, anderer Seed anderes Ergebnis
    s3 = waehle(d, dfb, gr, r1)
    assert s3['enden'].equals(s['enden']) and s3['paare'].equals(s['paare'])
    s4 = waehle(d, dfb, gr, r1, seed=SEED + 1)
    assert not s4['enden'].ansicht_nr.equals(s['enden'].ansicht_nr)
    ok += 1
    # 5. Mindestabstand der Enden eines Paars
    pos = s['enden'].pivot(index='paar_id', columns='ende', values='ansicht_nr')
    assert (np.abs(pos.a - pos.b) >= MIN_ABSTAND_ENDEN).all()
    ok += 1
    # 6. Geteiltes Ende (aufeinanderfolgende Paare) wird nur einmal gezeigt
    d2 = d.copy()
    i0, i1 = int(infl[0]), int(infl[1])
    d2.loc[i1, ['CSN', 'MRN']] = d2.loc[i0, ['CSN', 'MRN']].to_numpy()
    d2.loc[i1, ['c_a', 'w0_a', 'w1_a']] = d2.loc[i0, ['c_b', 'w0_b', 'w1_b']].to_numpy()
    s5 = waehle(d2, dfb, gr, r1b.iloc[0:0])
    assert s5['info']['geteilte_streifen'] >= 1 and s5['info']['streifen'] == s5['info']['enden'] - s5['info']['geteilte_streifen']
    ok += 1
    # 7. zu wenige Fuellpaare -> Fehler
    try:
        waehle(d.assign(split='test'), dfb, gr, r1)
        raise AssertionError('Fehler erwartet')
    except ValueError:
        ok += 1
    # 8. Mindestzahl der Fuellstreifen steht in der Auswahl (Regel: ebenso viele)
    assert inf['streifen'] == 4 * len(infl) - inf['geteilte_streifen']
    ok += 1
    # 9. Seite: Ersetzungen, Rundlauf, keine Kennungen, Vorlage wiederhergestellt
    alt = AB.HTML_VORLAGE
    x = np.sin(np.linspace(0, 20, AB.N_SAMPLES)) * 1.3
    ans = [{'nr': k + 1, **AB.kodiere(x * (k + 1))} for k in range(3)]
    h = baue_seite(ans, 'ab' * 32)
    assert AB.HTML_VORLAGE is alt
    assert 'referenzannotation_s3_einfluss_' in h and 'referenzannotation_s3_ergebnisse_' not in h
    assert h.count('s3\\\\annotation_einfluss\\\\') == 2 and 's3\\\\annotation\\\\' not in h
    assert 'nachträgliche Prüfung' in h and 's3einfl_' + ('ab' * 32)[:16] + '_' + VERSION in h and "VERSION='annot-einfluss-v1'" in h
    assert '\u2014' not in h and '\u2013' not in h and 'https://' not in h and 'http://' not in h.replace('xmlns', '') and '<script src' not in h
    assert 'einflussreich' not in h.lower()
    u = np.frombuffer(__import__('base64').b64decode(ans[0]['b64']), '<i2')
    assert np.max(np.abs(u / 1000.0 - x)) <= 0.0006
    ok += 1
    # 10. Test mit der Vorlage anderer Version: fehlende Stelle wird erkannt
    try:
        vorlage_einfluss(alt.replace('<div id="banners"></div>', '<div id="x"></div>'))
        raise AssertionError('Fehler erwartet')
    except AssertionError as e:
        assert 'Vorlage:' in str(e)
    ok += 1
    # 11. Ordnung: Zurueckweisungsverfahren liefert Permutation, Fehler bei Unmoeglichem
    perm = ordne(44, [(2 * i, 2 * i + 1) for i in range(22)], seed=1, min_abstand=8)
    assert sorted(perm.tolist()) == list(range(44))
    try:
        ordne(10, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)], seed=1, min_abstand=9, versuche=2000)
        raise AssertionError('Fehler erwartet')
    except ValueError:
        pass
    ok += 1
    # 12. Hash der Streifenliste haengt an der Reihenfolge
    sA = pd.DataFrame({'ansicht_nr': [1, 2], 'CSN': [5, 6], 'c': [1.0, 2.0], 'wiederholung_von': 0})
    assert AB.auswahl_hash(sA) != AB.auswahl_hash(sA.assign(ansicht_nr=[2, 1]))
    ok += 1
    print(f'Selbsttest {ok} von {ok} bestanden')


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('modus', choices=['selbsttest', 'auswahl', 'seite', 'kontrolle', 'testseite'])
    ap.add_argument('--ausgabe', default=None)
    a = ap.parse_args(argv)
    if a.modus == 'selbsttest':
        selbsttest()
    elif a.modus == 'testseite':
        modus_testseite(a.ausgabe)
    elif a.modus == 'auswahl':
        modus_auswahl()
    elif a.modus == 'seite':
        modus_seite()
    else:
        modus_kontrolle()


if __name__ == '__main__':
    main(sys.argv[1:])
