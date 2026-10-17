"""S3/N3-M MC-MED v2: Elektrolytarbeit (Kalium innerhalb der Person, Hyperkaliaemie und Bikarbonat), Spezifikation SPEZIFIKATION_S3_MCMED_v2.md.

GERUEST. Noch nicht auf echten Daten gelaufen. Alle Funktionen sind an Kunstdaten getestet (Modus selbsttest), einschliesslich eines Ende-zu-Ende-Tests mit
synthetischen CSV- und WFDB-Dateien. Es gibt genau EINE Auswertung (Modus gesamtlauf) ueber alle MC-MED-Splits (Train, Val, Test). Sie ist nur nach der Bindung
(spec_lock_s3_v2.json) moeglich; jede Datenfunktion verlangt eine Freigabe, die nur pruefe_start nach erfolgreicher Pruefung der Hashes ausstellt. Einen
Bestaetigungsanspruch gibt es nicht; Ergebnisse je Split sind nur ergaenzende Stabilitaetsbeschreibung.

Modi (Python: <local path> -I):
  selbsttest                     Kunstdaten-Tests, keine echten Daten, keine Sperre
  sperren                        spec_lock_s3_v2.json schreiben (nur K): Hashes von Spezifikation, Skript, Messkette, Konfiguration, Vertrag
  gesamtlauf [--arbeiter N]      die eine Gesamtauswertung (laden, extrahieren, Beschreibungen, Kontrollen, Schaetzer, Ausgabe)
  annotation                     kaliumblinde Referenzannotation vorbereiten (optional, vor dem Gesamtlauf)
Regeln: keine Kennungen in Ausgaben; Kleinzellenregel (1 bis 9 -> "<10") nach Zaehlfeldern; DuckDB 700 MB, 2 Threads; Einzelprozess unter 1,5 GB; keine Loeschbefehle.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path
import sys, os, json, time, copy, hashlib, math, re, tempfile, warnings, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
import numpy as np
import pandas as pd
import scipy.signal as ss

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
N3_DIR = Path(str(data_path("s3.n3_dir")))
SPEC_PATH = Path(str(data_path("s3.specification_v2_md")))
DATA = Path(str(data_path("s3.mcmed_data_dir")))
WAVE = DATA / 'waveforms'
PRIV = Path(str(data_path("s3.work_dir")))
LOCK = PRIV / 'spec_lock_s3_v2.json'
AUS = PRIV / 'auswertung_v2'
PIPELINE = 'aw255-v3'
FS = 500

# Zeit- und Einheitenvertrag (Spezifikation 3.2); Teil der Bindung und im Lauf geprueft
VERTRAG = {
    'zeit': 'UTC-Sekunden seit 1970-01-01; CSV-Zeitstempel (Suffix Z) als UTC, WFDB-Kopfzeit (Uhrzeit, Datum) als UTC gelesen',
    'signal': 'Ableitung II in mV (Kopfzeile units muss mV sein)',
    'kalium': 'mmol/L', 'bikarbonat': 'mmol/L', 'natrium': 'mmol/L', 'hf': '1/min',
    'zeitpruefung': 'Median(erster Monitor-HF-Zeitstempel minus erster Segmentbeginn) zwischen -60 und 300 s',
}

# --------------------------------------------------------------------------- Konfiguration (Spezifikation Abschnitte 4 bis 10)
CFG = {
    'version': 's3-mcmed-v2',
    'seeds': {'bootstrap': 255001, 'nk1': 255002, 'nk2': 255003, 'auc': 255005, 'pk': 255006, 'pksample': 255008, 'nk1_block': 255009, 'annotation': 255010,
              'split': 255011},
    'bootstrap_n': 2000, 'nk2_n': 500,
    'paar': {'gap_h_min': 1.0, 'gap_h_max': 12.0},
    'behandlung': {'vor_dok_s': 7200, 'vor_unklar_s': 21600},
    'fenster': {'puffer_s': 900, 'cap_nach_s': 10800, 'block_s': 10, 'raster_s': 5, 'messfenster_s': 60, 'rep_abstand_s': 90,
                'shift_min_s': 7200, 'shift_max_s': 21600, 'shift_versuche': 20, 'hf_halbfenster_s': 300, 'hf_min_werte': 3},
    'messkette': {'min_cover_s': 20.0, 'hf_schwelle': 0.0789531424618521, 'polaritaet_unsicher_ratio': 0.70},
    'primaer': {'erst_k_max': 6.0},
    'min_schaetzung': {'paare': 30, 'personen': 10},
    'hf_spline': {'knoten': [55.0, 70.0, 85.0, 110.0], 'grenzen': [30.0, 220.0]},
    'h1': {'grenze': 0.010},
    'h2': {'grenze': 0.2},
    'h3': {'k_min': 6.5, 'grenze': 0.29, 'n_min': 30},
    'h4': {'k_erst_min': 6.0, 'rep_min_h': 0.25, 'rep_max_h': 6.0, 'pe_min': 5.5, 'rn_max': 5.0, 'rn_abfall': 1.0, 'n_min_gruppe': 30,
           'schwelle_t_mv': 0.30, 'auc_marke': 0.65},
    'klinisch': {'mm_je_mv': 10.0, 'dk_strata': [-1.0, -0.5, 0.5, 1.0], 'rc_z': 1.96},
    'na': {'schritt': 5.0, 'grenze': 0.005, 'korr_max': 0.30},
    'nk': {'grenze': 0.005, 'n_min': 10},
    'messfehler': {'sd_k': [0.1, 0.2]},
    'annotation': {'n': 100},
    'kaliumsenkend': ['insulin_regular', 'glukose_bolus', 'bikarbonat_iv', 'albuterol', 'k_binder'],
    'ecg_wirksam': ['calcium_iv'],
    'dok_klassen': ['insulin_regular', 'glukose_bolus', 'bikarbonat_iv', 'albuterol', 'k_binder', 'calcium_iv', 'kalium_gabe', 'dialyse'],
    'unklar_klassen': ['insulin_andere', 'schleifendiuretikum'],
}
ZAEHL = re.compile(r'(^n$|^n_|_n$|_n_|^anzahl|paare|personen|entnahmen|besuche|fenster|^pe$|^rn$|^unklar|ausgeschlossen|zaehl|stratum|gruppe|messungen|bloecke)', re.I)


# =========================================================================== Maske nach Zaehlfeldern
def _ist_zaehl(key):
    return key is not None and bool(ZAEHL.search(str(key)))


def mask_tree(o, key=None):
    """Kleinzellenregel nach Zaehlfeldern: Werte 1 bis 9 unter Schluesseln, die Zaehlungen bezeichnen (auch als Gleitkomma), werden zu "<10".
    Andere Zahlen (Seeds, Schaetzer, Konfiguration) bleiben unveraendert."""
    if isinstance(o, dict):
        return {k: mask_tree(v, k) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [mask_tree(v, key) if not isinstance(v, (list, tuple, dict)) and _ist_zaehl(key) else mask_tree(v, None) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer, float, np.floating)):
        v = float(o)
        if not math.isfinite(v):
            return None
        if _ist_zaehl(key) and v == int(v) and 1 <= int(v) <= 9:
            return '<10'
        return int(o) if isinstance(o, (int, np.integer)) else v
    return o


def zaehlkette_vergroebern(z):
    """Komplemente vergroebern: Ist der Abgang zu einem Schritt 1 bis 9 (oder bleibt ein Schritt nach einem vergroeberten gleich), wird der Wert auf Zehner abgerundet
    und markiert, damit die gerundete Zahl nicht ueber die Folgeschritte rekonstruierbar ist."""
    out, prev, prev_grob, grob = {}, None, False, []
    for k, v in z.items():
        if isinstance(v, (int, np.integer)):
            d = None if prev is None else prev - int(v)
            runden = d is not None and (1 <= d <= 9 or (d == 0 and prev_grob))
            out[k] = (int(v) // 10) * 10 if runden else int(v)
            if runden:
                grob.append(k)
            prev, prev_grob = int(v), runden
        else:
            out[k] = v
    if grob:
        out['gerundet_auf_zehner'] = grob
    return out


def sha(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


# =========================================================================== Bindung, Freigabe
_TOKEN = object()
ABHAENGIG = [HIER / 's3_signal.py', N3_DIR / 'aw197_messung.py', N3_DIR / 'aw203_messung.py']


class Freigabe:
    """Wird nur von pruefe_start ausgestellt; jede Datenfunktion verlangt sie (Testsperre in jeder Funktion)."""
    __slots__ = ('hash',)

    def __init__(self, token, h):
        if token is not _TOKEN:
            raise PermissionError('Freigabe nur ueber pruefe_start nach erfolgreicher Bindungspruefung')
        self.hash = h


def pflicht_freigabe(f):
    if not isinstance(f, Freigabe):
        raise PermissionError('Datenzugriff ohne Freigabe (Bindung nicht geprueft)')
    return f


def hashes():
    return {'spezifikation': sha(SPEC_PATH) if SPEC_PATH.exists() else None, 'skript': sha(Path(__file__)),
            'abhaengig': {p.name: sha(p) for p in ABHAENGIG if p.exists()},
            'konfiguration': hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest(),
            'vertrag': hashlib.sha256(json.dumps(VERTRAG, sort_keys=True).encode()).hexdigest(), 'pipeline': PIPELINE}


def modus_sperren():
    if LOCK.exists():
        raise SystemExit('Bindung existiert bereits: ' + str(LOCK))
    h = hashes()
    if not h['spezifikation']:
        raise SystemExit('Spezifikation fehlt')
    PRIV.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({'gebunden': dt.datetime.now(dt.timezone.utc).isoformat(), 'hashes': h, 'cfg': CFG, 'vertrag': VERTRAG}, indent=1, ensure_ascii=False), encoding='utf-8')
    print('gebunden', LOCK)


def pruefe_start(lock=None, hash_fn=None):
    """Hash der Spezifikation, des Skripts, der Messkette, der Konfiguration und des Vertrags muessen der Bindung entsprechen. Rueckgabe: Freigabe."""
    lock = Path(lock or LOCK)
    if not lock.exists():
        raise SystemExit('nicht gebunden: ' + str(lock))
    soll = json.loads(lock.read_text(encoding='utf-8'))['hashes']
    ist = (hash_fn or hashes)()
    for k in ('spezifikation', 'skript', 'konfiguration', 'vertrag', 'pipeline'):
        if soll.get(k) != ist.get(k):
            raise SystemExit(f'Abbruch: {k} weicht von der Bindung ab')
    if soll.get('abhaengig') != ist.get('abhaengig'):
        raise SystemExit('Abbruch: abhaengige Messkette weicht von der Bindung ab')
    return Freigabe(_TOKEN, sha(lock))


# =========================================================================== Behandlung
DIALYSE_RE = re.compile(r'DIALYZE|HEMODIALYSIS|CRRT|PERIT\w*\.? ?DIALYSIS|PERI\.DIALYSIS|CAPD|CCPD|DIALYSATE|HEMODIAL SOLN|DIALY NO', re.I)


def behandlung_klasse(name):
    """Klassifikation von Auftraegen (Spezifikation 4.2). Rueckgabe None, wenn nicht relevant."""
    u = str(name).upper()
    if 'DIALYSIS ACCESS' not in u and 'CONSULT' not in u and DIALYSE_RE.search(u):
        return 'dialyse'
    if re.search(r'INSULIN REGULAR|INSULIN, REGULAR', u):
        return 'insulin_regular'
    if 'INSULIN' in u:
        return 'insulin_andere'
    if re.search(r'DEXTROSE 50|D50W|DEXTROSE 20|D20W', u):
        return 'glukose_bolus'
    if re.search(r'CALCIUM GLUC|CALCIUM CHLORIDE', u):
        return 'calcium_iv'
    if re.search(r'SODIUM BICARBONATE (8\.4|IVPB)|BICARBONATE IV|D5W WITH SODIUM BICARBONATE', u):
        return 'bikarbonat_iv'
    if 'ALBUTEROL' in u:
        return 'albuterol'
    if re.search(r'POLYSTYRENE|ZIRCONIUM|PATIROMER|LOKELMA', u):
        return 'k_binder'
    if re.search(r'FUROSEMIDE|BUMETANIDE|TORSEMIDE', u):
        return 'schleifendiuretikum'
    if re.search(r'POTASSIUM CHLORIDE|POTASSIUM PHOSPHATE|POTASSIUM, SODIUM PHOSPHATES|POTASSIUM CHLORID', u):
        return 'kalium_gabe'
    if 'MAGNESIUM SULFATE' in u:
        return 'magnesium_iv'
    return None


def behandlung_kategorie(P, trt, cfg=CFG):
    """Drei Kategorien je Paar (Spezifikation 4.2): dokumentiert / unklar / kein_hinweis.
    Fenster [S, E]: S = min(ta, ECG-Mitte a), E = max(tb, Ergebniszeit b, ECG-Mitte b). Ränder einschliesslich (>=, <=).
    dokumentiert: Gabe (First_admin_time) einer dok_klasse in [S - 120 min, E]; Dialyse-Auftrag in [S - 360 min, E].
    unklar: sonst Gabe einer dok_klasse in [S - 360 min, S - 120 min); Auftrag einer dok_klasse ohne Gabezeit in [S - 120 min, E]; Gabe von insulin_andere oder
    schleifendiuretikum in [S - 120 min, E]; Dialyse-Auftrag irgendwo im Besuch ausserhalb des Fensters.
    trt: CSN, cls, t_admin (NaN ohne Gabezeit), t_ord. P: CSN, ta, tb, rb, c_a, c_b (NaN erlaubt)."""
    b = cfg['behandlung']
    dok_k = set(cfg['dok_klassen'])
    unk_k = set(cfg['unklar_klassen'])
    tg = {k: v for k, v in trt.groupby('CSN')} if len(trt) else {}
    kat, nkl, nca, nkg, ndi = [], [], [], [], []
    for r in P.itertuples():
        S = min(r.ta, r.c_a) if np.isfinite(getattr(r, 'c_a', np.nan)) else r.ta
        ends = [r.tb] + ([r.rb] if np.isfinite(getattr(r, 'rb', np.nan)) else []) + ([r.c_b] if np.isfinite(getattr(r, 'c_b', np.nan)) else [])
        E = max(ends)
        g = tg.get(r.CSN)
        k = 'kein_hinweis'
        c_kl = c_ca = c_kg = c_di = 0
        if g is not None:
            gab = g[g.t_admin.notna()]
            od = g[g.t_admin.isna()]
            dok = gab[gab.cls.isin(dok_k) & (gab.t_admin >= S - b['vor_dok_s']) & (gab.t_admin <= E)]
            c_kl = int(dok.cls.isin(cfg['kaliumsenkend']).sum())
            c_ca = int((dok.cls == 'calcium_iv').sum())
            c_kg = int((dok.cls == 'kalium_gabe').sum())
            dial_ord = g[(g.cls == 'dialyse') & (g.t_ord >= S - b['vor_unklar_s']) & (g.t_ord <= E)]
            c_di = int(len(dial_ord)) + int(((dok.cls == 'dialyse')).sum())
            documented = len(dok) > 0 or len(dial_ord) > 0
            if documented:
                k = 'dokumentiert'
            else:
                frueh = gab[gab.cls.isin(dok_k) & (gab.t_admin >= S - b['vor_unklar_s']) & (gab.t_admin < S - b['vor_dok_s'])]
                ohne = od[od.cls.isin(dok_k - {'dialyse'}) & (od.t_ord >= S - b['vor_dok_s']) & (od.t_ord <= E)]
                schwach = gab[gab.cls.isin(unk_k) & (gab.t_admin >= S - b['vor_dok_s']) & (gab.t_admin <= E)]
                dial_sonst = g[(g.cls == 'dialyse')]
                if len(frueh) or len(ohne) or len(schwach) or len(dial_sonst):
                    k = 'unklar'
        kat.append(k)
        nkl.append(c_kl)
        nca.append(c_ca)
        nkg.append(c_kg)
        ndi.append(c_di)
    out = P.copy()
    out['beh_kat'] = kat
    out['n_kl'] = nkl
    out['n_ca'] = nca
    out['n_kgabe'] = nkg
    out['n_dialyse'] = ndi
    return out


# =========================================================================== Statistik
def rcs_basis(x, knoten, grenzen=None):
    """Eingeschraenkte kubische Splines (Harrell, natuerlich) mit festen Knoten: Spalten x, und k-2 nichtlineare Terme. Werte werden auf die Grenzen begrenzt."""
    x = np.asarray(x, float)
    if grenzen is not None:
        x = np.clip(x, grenzen[0], grenzen[1])
    t = np.asarray(knoten, float)
    k = len(t)
    cols = [x]
    tk, tk1 = t[-1], t[-2]
    pos = lambda u: np.maximum(u, 0.0) ** 3
    for j in range(k - 2):
        c = pos(x - t[j]) - pos(x - tk1) * (tk - t[j]) / (tk - tk1) + pos(x - tk) * (tk1 - t[j]) / (tk - tk1)
        cols.append(c / (tk - t[0]) ** 2)
    return np.column_stack(cols)


def spline_differenz(hf_a, hf_b, cfg=CFG):
    """f(HF2) - f(HF1) mit f = eingeschraenkter kubischer Spline mit festen Knoten; Rueckgabe Array (n, k-1) mit Spalten dB1..dBk-1."""
    s = cfg['hf_spline']
    return rcs_basis(hf_b, s['knoten'], s['grenzen']) - rcs_basis(hf_a, s['knoten'], s['grenzen'])


def ols(X, y):
    b, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)
    return b


def cluster_bootstrap(df, cluster, stat, B, seed):
    """Personen-Clusterbootstrap: Cluster mit Zuruecklegen, Zeilen aller gezogenen Cluster uebernehmen. stat(df) -> Vektor/Skalar. Rueckgabe Array (B, k)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(df[cluster].to_numpy(), return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    n = len(uniq)
    k = len(np.atleast_1d(stat(df)))
    res = []
    for _ in range(B):
        pick = rng.integers(0, n, n)
        idx = np.concatenate([groups[p] for p in pick])
        try:
            res.append(np.atleast_1d(stat(df.iloc[idx])))
        except Exception:
            res.append(np.full(k, np.nan))
    return np.array(res)


def ci_perz(boot, j=0, alpha=0.05):
    v = boot[:, j]
    v = v[np.isfinite(v)]
    if len(v) < 50:
        return (np.nan, np.nan, np.nan)
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(min(1.0, 2.0 * min((v <= 0).mean(), (v >= 0).mean())))


def steigung(df, y, x, kov=()):
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy()] + [df[k].to_numpy() for k in kov])
    return ols(X, df[y].to_numpy())[1]


def schaetze(df, y, x, kov=(), cfg=CFG, cluster='MRN', seed=None, B=None):
    """Steigung von x auf y (freier Achsenabschnitt) mit Personen-Clusterbootstrap. Rueckgabe dict (n, personen, beta, ci_lo, ci_hi, p_boot, beurteilbar)."""
    d = df.dropna(subset=[y, x] + list(kov)).copy()
    ms = cfg['min_schaetzung']
    if len(d) < ms['paare'] or d[cluster].nunique() < ms['personen']:
        return {'n': int(len(d)), 'personen': int(d[cluster].nunique()), 'beurteilbar': False}
    f = lambda t: steigung(t, y, x, kov)
    b = float(f(d))
    boot = cluster_bootstrap(d, cluster, lambda t: np.array([f(t)]), B or cfg['bootstrap_n'], seed if seed is not None else cfg['seeds']['bootstrap'])
    lo, hi, p = ci_perz(boot)
    return {'n': int(len(d)), 'personen': int(d[cluster].nunique()), 'beta': b, 'ci_lo': lo, 'ci_hi': hi, 'se_boot': float(np.nanstd(boot[:, 0], ddof=1)),
            'p_boot': p, 'beurteilbar': True}


def schaetze_huber(df, y, x, kov=(), cfg=CFG, cluster='MRN', seed=None):
    """Robuste Sensitivitaet: Huber-M-Schaetzer (statsmodels RLM) mit Personen-Clusterbootstrap."""
    import statsmodels.api as sm
    d = df.dropna(subset=[y, x] + list(kov)).copy()
    ms = cfg['min_schaetzung']
    if len(d) < ms['paare'] or d[cluster].nunique() < ms['personen']:
        return {'n': int(len(d)), 'beurteilbar': False}

    def f(t):
        X = np.column_stack([np.ones(len(t)), t[x].to_numpy()] + [t[k].to_numpy() for k in kov])
        return sm.RLM(t[y].to_numpy(), X, M=sm.robust.norms.HuberT()).fit().params[1]

    b = float(f(d))
    boot = cluster_bootstrap(d, cluster, lambda t: np.array([f(t)]), max(200, cfg['bootstrap_n'] // 4), (seed if seed is not None else cfg['seeds']['bootstrap']) + 7)
    lo, hi, p = ci_perz(boot)
    return {'n': int(len(d)), 'beta': b, 'ci_lo': lo, 'ci_hi': hi, 'beurteilbar': True}


def einfluss_pruefung(df, y, x, kov=(), cfg=CFG, cluster='MRN'):
    """Einflussreiche Paare: |DFBETAS| der Steigung > 2/sqrt(n); Schaetzer ohne diese Paare (Sensitivitaet), Zahl und groesster Wert."""
    from statsmodels.stats.outliers_influence import OLSInfluence
    import statsmodels.api as sm
    d = df.dropna(subset=[y, x] + list(kov)).reset_index(drop=True)
    if len(d) < cfg['min_schaetzung']['paare']:
        return {'beurteilbar': False, 'n': int(len(d))}
    X = np.column_stack([np.ones(len(d)), d[x].to_numpy()] + [d[k].to_numpy() for k in kov])
    m = sm.OLS(d[y].to_numpy(), X).fit()
    dfb = np.abs(OLSInfluence(m).dfbetas[:, 1])
    grenze = 2.0 / math.sqrt(len(d))
    keep = dfb <= grenze
    out = {'beurteilbar': True, 'n': int(len(d)), 'n_einflussreich': int((~keep).sum()), 'dfbetas_max': float(dfb.max()), 'grenze': grenze}
    out['ohne_einflussreiche'] = schaetze(d[keep], y, x, kov, cfg, cluster, B=max(200, cfg['bootstrap_n'] // 4))
    return out


def gemischtes_modell(df, y, x, kov=(), beta_ref=None, cluster='MRN'):
    """Sensitivitaet: lineares gemischtes Modell (zufaelliger Achsenabschnitt je Person, REML). Kovariaten werden standardisiert (Steigung von x bleibt unveraendert);
    Optimierer lbfgs, bfgs, powell nacheinander. Konvergenz und Vorzeichenuebereinstimmung mit beta_ref werden geprueft; Warnungen und Fehler werden protokolliert
    statt unterdrueckt. Scheitern alle Optimierer, ist das Modell nicht beurteilbar (Sensitivitaet entfaellt)."""
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[y, x] + list(kov)).copy()
    if len(d) < 20 or d[cluster].nunique() < 10:
        return {'n': int(len(d)), 'beurteilbar': False}
    for k in kov:
        sd = d[k].std()
        d[k] = (d[k] - d[k].mean()) / (sd if sd > 0 else 1.0)
    f = f"{y} ~ {x}" + ''.join(f" + {k}" for k in kov)
    fehler, warn = [], set()
    for meth in ('lbfgs', 'bfgs', 'powell'):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                m = smf.mixedlm(f, d, groups=d[cluster]).fit(reml=True, method=meth)
                warn |= {str(i.message)[:120] for i in w}
            ci = m.conf_int().loc[x]
            out = {'n': int(len(d)), 'beta': float(m.params[x]), 'ci_lo': float(ci[0]), 'ci_hi': float(ci[1]), 'konvergiert': bool(m.converged), 'methode': meth,
                   'warnungen': sorted(warn), 'fehler_vorher': fehler, 'beurteilbar': True}
            out['vorzeichen_gleich'] = None if beta_ref is None else bool(np.sign(beta_ref) == np.sign(out['beta']))
            out['vertrauenswuerdig'] = bool(out['konvergiert'] and out['vorzeichen_gleich'] in (None, True))
            return out
        except Exception as e:
            fehler.append(f'{meth}: {type(e).__name__}')
    return {'n': int(len(d)), 'beurteilbar': False, 'konvergiert': False, 'warnungen': sorted(warn), 'fehler': fehler, 'vertrauenswuerdig': False}


def kategorie_intervall(est, grenze, erwartet=+1):
    """Ueberschneidungsfreie Kategorien (Spezifikation 6.1) fuer ein Intervall (lo, hi) gegen null und die Grenze G (Erwartung positiv; erwartet=-1 spiegelt):
    negativ                     hi < 0                       (entgegen der Erwartung; nach technischer Pruefung interpretierbar)
    positiv_ueber_grenze        lo >= G                      (Effekte unterhalb der Grenze ausgeschlossen)
    positiv_unter_grenze        0 < lo, hi < G               (nachweisbar, aber Effekte ab der Grenze ausgeschlossen)
    positiv_grenze_offen        0 < lo < G <= hi
    grenze_ausgeschlossen       lo <= 0, hi < G              (kein Nachweis; Effekte ab der Grenze ausgeschlossen)
    unpraezise                  lo <= 0, hi >= G
    nicht_beurteilbar           keine Schaetzung"""
    if not est.get('beurteilbar'):
        return 'nicht_beurteilbar'
    lo, hi = (est['ci_lo'], est['ci_hi']) if erwartet > 0 else (-est['ci_hi'], -est['ci_lo'])
    if hi < 0:
        return 'negativ'
    if lo > 0:
        if lo >= grenze:
            return 'positiv_ueber_grenze'
        if hi < grenze:
            return 'positiv_unter_grenze'
        return 'positiv_grenze_offen'
    return 'grenze_ausgeschlossen' if hi < grenze else 'unpraezise'


def kategorie_nullnaehe(est, margin):
    """Nullnaehe nur ueber Aequivalenzgrenzen: nahe_null (Intervall innerhalb +-margin), von_null_verschieden (Intervall schliesst null aus), sonst unpraezise.
    Ein Intervall, das null enthaelt, gilt allein nicht als Nullnaehe."""
    if not est.get('beurteilbar'):
        return 'nicht_beurteilbar'
    if est['ci_lo'] >= -margin and est['ci_hi'] <= margin:
        return 'nahe_null'
    if est['ci_lo'] > 0 or est['ci_hi'] < 0:
        return 'von_null_verschieden'
    return 'unpraezise'


def icc_zweifenster(a, b):
    """Wiederholbarkeit aus zwei Messungen derselben Entnahme: SD innerhalb, ICC(1,1), Bias, Wiederholungsgrenze RC = 1,96*sqrt(2)*SD innerhalb."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 10:
        return {'n': int(n), 'beurteilbar': False}
    d = a - b
    sw = float(np.std(d, ddof=1) / np.sqrt(2))
    m = (a + b) / 2
    msb = 2 * np.var(m, ddof=1)
    msw = float(np.sum(d ** 2) / (2 * n))
    return {'n': int(n), 'beurteilbar': True, 'sw': sw, 'icc': float((msb - msw) / (msb + msw)), 'bias': float(d.mean()),
            'rc': float(1.96 * math.sqrt(2) * sw), 'loa_halb': float(1.96 * np.std(d, ddof=1))}


def auc_mw(x1, x0):
    x1 = np.asarray(x1, float)
    x0 = np.asarray(x0, float)
    if len(x1) == 0 or len(x0) == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([x1, x0]))
    u = r[:len(x1)].sum() - len(x1) * (len(x1) + 1) / 2.0
    return float(u / (len(x1) * len(x0)))


def mm(mv, cfg=CFG):
    """mV in mm bei Standardverstaerkung (10 mm/mV)."""
    return None if mv is None else float(mv) * cfg['klinisch']['mm_je_mv']


# =========================================================================== Signal: Lesen, Polaritaet, Bloecke, Messung
_SG = None
_CHAIN = None


def sg():
    global _SG
    if _SG is None:
        import s3_signal as S
        _SG = S
    return _SG


def chain():
    """Messkette AW-203 (unveraendert, gebunden) laden; Mindestabdeckung gueltiger Schlaege fuer 60-s-Fenster auf 20 s (Spezifikation 5.1)."""
    global _CHAIN
    if _CHAIN is None:
        if str(N3_DIR) not in sys.path:
            sys.path.insert(0, str(N3_DIR))
        import aw197_messung as M
        import aw203_messung as V
        M.CFG['min_cover_s'] = CFG['messkette']['min_cover_s']
        _CHAIN = (M, V)
    return _CHAIN


def einheit_pruefen(units):
    """Einheitenvertrag: Ableitung II muss in mV vorliegen."""
    u = (units[0] if units else '') or ''
    if str(u).strip().lower() != 'mv':
        raise ValueError(f'Einheit {u!r} statt mV')
    return True


def lies_roh(csn, seg, a_t, b_t, wave=None):
    """Roh-EKG II (mV) eines Segments von a_t bis b_t (absolute Sekunden, an die Segmentgrenzen begrenzt). seg = (nr, fs, n, t0, t1).
    Rueckgabe (Startzeit absolut, Array) oder None bei weniger als 10 s."""
    import wfdb
    sn, fs, n, t0, t1 = seg
    a = max(0, int(np.floor((a_t - t0) * fs)))
    b = min(int(n), int(np.ceil((b_t - t0) * fs)))
    if b - a < 10 * fs:
        return None
    name = str(Path(wave or WAVE) / str(csn)[-3:] / str(csn) / 'II' / f'{csn}_{sn}')
    rec = wfdb.rdrecord(name, sampfrom=a, sampto=b, physical=True, channels=[0])
    einheit_pruefen(rec.units)
    return t0 + a / fs, rec.p_signal[:, 0].astype(float)


def polaritaet(x, fs=FS, cfg=CFG):
    """Polaritaet der QRS-Hauptrichtung aus dem Medianschlag. Rueckgabe (pol, unsicher, n_schlaege); pol = +1 oder -1, None bei zu wenig Schlaegen.
    unsicher, wenn die kleinere Auslenkung mindestens `polaritaet_unsicher_ratio` der groesseren betraegt."""
    x = np.nan_to_num(np.asarray(x, float))
    if len(x) < int(5 * fs):
        return None, True, 0
    xf = ss.sosfiltfilt(ss.butter(2, [0.5, 40], 'bp', fs=fs, output='sos'), x)
    xq = ss.sosfiltfilt(ss.butter(2, [5, 20], 'bp', fs=fs, output='sos'), x)
    env = np.convolve(np.gradient(xq) ** 2, np.ones(int(0.1 * fs)) / int(0.1 * fs), 'same')
    pk, _ = ss.find_peaks(env, height=0.35 * np.percentile(env, 98), distance=int(0.25 * fs))
    a = int(0.04 * fs)
    ctr = []
    for p in pk:
        lo, hi = max(0, p - a), min(len(xf), p + a + 1)
        ctr.append(lo + int(np.argmax(np.abs(xf[lo:hi]))))
    pre = post = int(0.10 * fs)
    ok = [r for r in ctr if r - pre >= 0 and r + post <= len(xf)]
    if len(ok) < 5:
        return None, True, len(ok)
    med = np.median(np.array([xf[r - pre:r + post] for r in ok]), axis=0)
    base = float(np.median(np.concatenate([med[:int(0.02 * fs)], med[-int(0.02 * fs):]])))
    apos, aneg = float(med.max() - base), float(base - med.min())
    if apos <= 0 and aneg <= 0:
        return None, True, len(ok)
    gross, klein = max(apos, aneg), min(apos, aneg)
    return (1 if apos >= aneg else -1), bool(gross <= 0 or klein / gross >= cfg['messkette']['polaritaet_unsicher_ratio']), len(ok)


def messe_fenster(x, ts, t0w, t1w, cfg=CFG):
    """Messkette aw255-v3: Polaritaet bestimmen, Signal so drehen, dass die QRS-Hauptrichtung positiv ist, dann AW-203 v2 (measure_window_v2).
    Endpunkt: QRS-RELATIVE T-Amplitude (t_amp, positiv = T gleichgerichtet zum QRS); t_roh = rohe vorzeichenbehaftete Ableitung-II-Amplitude (pol * t_amp);
    steil = -t_slope des absteigenden T-Schenkels. x: Roh-EKG ab Zeit ts, Fenster [t0w, t1w]."""
    M, V = chain()
    i0, i1 = int(round((t0w - ts) * FS)), int(round((t1w - ts) * FS))
    out = {'valid': 0, 'reason': '', 'pol': np.nan, 'pol_unsicher': 1, 't_amp': np.nan, 't_roh': np.nan, 'steil': np.nan, 'qrs_ms': np.nan, 'pr_ms': np.nan, 'hr': np.nan}
    if i0 < 0 or i1 > len(x) or i1 - i0 < 20 * FS:
        out['reason'] = 'kein_signal'
        return out
    pol, uns, nb = polaritaet(np.asarray(x[i0:i1], float), FS, cfg)
    if pol is None:
        out['reason'] = 'polaritaet_unbestimmt'
        return out
    out['pol'], out['pol_unsicher'] = pol, int(uns)
    r = V.measure_window_v2(pol * np.asarray(x, float), ts, t0w, t1w, cfg['messkette']['hf_schwelle'], splithalf=False, do_dwt=True)
    out['valid'] = int(r.get('valid', 0))
    out['reason'] = r.get('reason', '') or ''
    for k in ('t_amp', 'qrs_ms', 'pr_ms', 'hr'):
        out[k] = r.get(k, np.nan)
    out['t_roh'] = pol * out['t_amp'] if out['t_amp'] == out['t_amp'] else np.nan
    ts_ = r.get('t_slope', np.nan)
    out['steil'] = -ts_ if ts_ == ts_ else np.nan
    return out


def saubere_bloecke_nach_abstand(xr, ts, t_ref, lo, hi, cfg=CFG, fs=FS):
    """Saubere 10-s-Bloecke (Guete s3_signal.block_gut) im Fenster [lo, hi] (absolut) in aufsteigendem Abstand der Blockmitte von t_ref, Raster 5 s.
    Rueckgabe: Generator (mitte, sqi, hf)."""
    S = sg()
    xr = np.asarray(xr, float)
    nanm = ~np.isfinite(xr)
    if len(xr) < 200:
        return
    x0 = np.where(nanm, 0.0, xr)
    xf = ss.sosfiltfilt(S.SOS_BP, x0)
    xq = ss.sosfiltfilt(S.SOS_QRS, x0)
    L = cfg['fenster']['block_s'] * fs
    a, b = max(ts, lo), min(ts + len(xr) / fs, hi)
    if b - a < cfg['fenster']['block_s']:
        return
    starts = np.arange(a, b - cfg['fenster']['block_s'] + 1e-9, cfg['fenster']['raster_s'])
    mitten = starts + cfg['fenster']['block_s'] / 2.0
    for k in np.argsort(np.abs(mitten - t_ref), kind='stable'):
        i = int(round((starts[k] - ts) * fs))
        if i < 0 or i + L > len(xr) or nanm[i:i + L].any():
            continue
        gut, sqi, hrb, nb, grund = S.block_gut(xr[i:i + L], xf[i:i + L], xq[i:i + L])
        if gut:
            yield float(mitten[k]), float(sqi), float(hrb)


def messfenster_um(c, ts, te, cfg=CFG):
    """60-s-Messfenster um die Blockmitte c, in die tatsaechlichen Segmentgrenzen [ts, te] verschoben. None, wenn das Segment kuerzer als das Fenster ist."""
    w = cfg['fenster']['messfenster_s']
    if te - ts < w:
        return None
    t0w = min(max(c - w / 2.0, ts), te - w)
    return t0w, t0w + w


def naechster_block(csn, segs, t_ref, lo, hi, ausser=(), min_abstand=0.0, cfg=CFG, wave=None):
    """Globale Suche ueber ALLE Segmente, die [lo, hi] schneiden: sauberer Block mit dem kleinsten Abstand der Mitte zu t_ref; Bloecke, deren Mitte weniger als
    min_abstand von einer Mitte in `ausser` entfernt ist, werden uebersprungen (absolute Zeit, auch ueber Segmente). Rueckgabe (abstand, mitte, seg) oder None."""
    best = None
    for seg in segs:
        sn, fs, n, t0, t1 = seg
        if fs != FS or t1 < lo or t0 > hi:
            continue
        try:
            r = lies_roh(csn, seg, max(lo, t0), min(hi, t1), wave)
        except Exception:
            continue
        if r is None:
            continue
        ts_, x = r
        for c, sqi, hrb in saubere_bloecke_nach_abstand(x, ts_, t_ref, lo, hi, cfg):
            if any(abs(c - e) < min_abstand for e in ausser):
                continue
            d = abs(c - t_ref)
            if best is None or d < best[0]:
                best = (d, c, seg)
            break
    return best


def messe_bei(csn, seg, c, cfg=CFG, wave=None):
    """Messung im 60-s-Fenster um die Blockmitte c; das Fenster wird an die Segmentgrenzen verschoben und frisch aus dem Segment gelesen.
    Rueckgabe dict mit Messwerten und Fensterrand (w0, w1) oder None."""
    sn, fs, n, t0, t1 = seg
    fe = messfenster_um(c, t0, t1, cfg)
    if fe is None:
        return None
    pad = 4.0
    r = lies_roh(csn, seg, fe[0] - pad, fe[1] + pad, wave)
    if r is None:
        return None
    ts_, x = r
    m = messe_fenster(x, ts_, fe[0], fe[1], cfg)
    m['w0'], m['w1'], m['c'] = fe[0], fe[1], c
    return m


def d3_grenzen(t_order, t_result, cfg=CFG):
    """D3 = [fruehere Zeit - 15 min, min(spaetere Zeit + 15 min, fruehere Zeit + 3 h)]. Ohne Ergebniszeit gibt es kein D3 (None)."""
    if t_result is None or not np.isfinite(t_result):
        return None
    fw = cfg['fenster']
    fr, sp = min(t_order, t_result), max(t_order, t_result)
    return fr - fw['puffer_s'], min(sp + fw['puffer_s'], fr + fw['cap_nach_s'])


def shift_gueltig(c, w0, w1, t_order, lo, hi, cfg=CFG):
    """NK1: der TATSAECHLICH gewaehlte Block muss 2 bis 6 h von der Auftragszeit entfernt liegen und sein Messfenster muss ausserhalb von D3 liegen."""
    fw = cfg['fenster']
    return bool((fw['shift_min_s'] <= abs(c - t_order) <= fw['shift_max_s']) and (w1 < lo or w0 > hi))


def messe_entnahme(args):
    """Messung einer Entnahme (Spezifikation 5): D3-Hauptfenster (global naechster sauberer Block), D1 (eigene globale Suche), zweites Fenster (Wiederholbarkeit,
    absolute Mitten-Abstand >= rep_abstand_s), zeitversetztes Fenster (NK1, tatsaechlich gewaehlter Block geprueft). Ohne Ergebniszeit keine Messung.
    args = (csn, t_order, t_result, segs, flags)."""
    csn, t_order, t_result, segs, flags = args
    cfg = flags.get('cfg', CFG)
    wave = flags.get('wave')
    fw = cfg['fenster']
    out = {'CSN': csn, 't_order': t_order, 'ohne_ergebnis': 0}
    g = d3_grenzen(t_order, t_result, cfg)
    if g is None:
        out['ohne_ergebnis'] = 1
        out['blk'] = 0
        return out
    lo, hi = g
    out['abst_ergebnis_min'] = (t_result - t_order) / 60.0
    b3 = naechster_block(csn, segs, t_order, lo, hi, cfg=cfg, wave=wave)
    b1 = naechster_block(csn, segs, t_order, t_order - 900.0, t_order + 900.0, cfg=cfg, wave=wave)
    out['blk'] = int(b3 is not None)
    cache = {}

    def messen(b, suffix):
        if b is None:
            return None
        key = (b[2][0], round(b[1], 3))
        if key not in cache:
            cache[key] = messe_bei(csn, b[2], b[1], cfg, wave)
        m = cache[key]
        out['c_' + suffix] = b[1]
        if m is not None:
            for k, v in m.items():
                if k != 'c':
                    out[f'{k}_{suffix}'] = v
        return m

    m3 = messen(b3, 'd3')
    if b3 is not None:
        out['abst_auftrag_d3'] = b3[1] - t_order
        out['abst_ergebnis_d3'] = b3[1] - t_result
    messen(b1, 'd1')
    if flags.get('rep', True) and b3 is not None:
        br = naechster_block(csn, segs, t_order, lo, hi, ausser=[b3[1]], min_abstand=fw['rep_abstand_s'], cfg=cfg, wave=wave)
        messen(br, 'rep')
    if flags.get('shift', False):
        rng = np.random.default_rng([cfg['seeds']['nk1_block'], int(csn) % (2 ** 31), int(t_order) % (2 ** 31)])
        for _ in range(fw['shift_versuche']):
            cs = t_order + (1 if rng.random() < 0.5 else -1) * rng.uniform(fw['shift_min_s'], fw['shift_max_s'])
            bs = naechster_block(csn, segs, cs, cs - 30.0, cs + 30.0, cfg=cfg, wave=wave)
            if bs is None:
                continue
            m = messe_bei(csn, bs[2], bs[1], cfg, wave)
            if m is None:
                continue
            # tatsaechlich gewaehlter Block und Fenster pruefen: Abstand zur Auftragszeit im Bereich, Fenster ausserhalb von D3
            if shift_gueltig(bs[1], m['w0'], m['w1'], t_order, lo, hi, cfg):
                for k, v in m.items():
                    out[f'{k}_sh'] = v
                out['c_sh'] = bs[1]
                out['sh_ok'] = 1
                break
    return out


# =========================================================================== Paare
def paare_bauen(draws, cfg=CFG):
    """Konsekutive Paare je Besuch (naechste Entnahme nach Auftragszeit, Abstand 1 bis 12 h). draws: DataFrame (CSN, MRN, split, t_order, t_result, k, cens, age, sex,
    na, hco3, glc, ca). Flags: ohne_ergebnis (Ergebniszeit an einem Ende fehlt: Paar wird ausgeschlossen, nicht ergaenzt), zensiert (Kalium mit < oder >)."""
    d = draws.sort_values(['CSN', 't_order']).reset_index(drop=True)
    nxt = d.groupby('CSN').shift(-1)
    P = pd.DataFrame({'CSN': d.CSN, 'MRN': d.MRN, 'split': d.get('split', 'x'), 'ta': d.t_order, 'tb': nxt.t_order, 'ra': d.t_result, 'rb': nxt.t_result, 'ka': d.k, 'kb': nxt.k,
                      'zensiert': d.cens.astype(bool) | nxt.cens.fillna(False).astype(bool), 'age': d.age, 'sex': d.sex, 'na_a': d.na, 'na_b': nxt.na,
                      'hco3_a': d.hco3, 'hco3_b': nxt.hco3, 'glc_a': d.glc, 'glc_b': nxt.glc})
    P = P.dropna(subset=['tb']).copy()
    P['gap_h'] = (P.tb - P.ta) / 3600.0
    P = P[(P.gap_h >= cfg['paar']['gap_h_min']) & (P.gap_h <= cfg['paar']['gap_h_max'])].copy()
    P['ohne_ergebnis'] = P.ra.isna() | P.rb.isna()
    P['dk'] = P.kb - P.ka
    P['dna'] = P.na_b - P.na_a
    P['dhco3'] = P.hco3_b - P.hco3_a
    P['dglc'] = P.glc_b - P.glc_a
    return P.reset_index(drop=True)


def messwerte_anfuegen(P, mess, prefix, hf, trt, cfg=CFG):
    """Messwerte beider Enden (Schluessel CSN, t_order) anfuegen, ECG-Zeiten, Polaritaet, Fensterkollision, Behandlungskategorie, HF.
    hf: DataFrame (CSN, t_order, hf)."""
    cols = ['valid', 'pol', 'pol_unsicher', 't_amp', 't_roh', 'steil', 'qrs_ms', 'c', 'w0', 'w1', 'reason']
    mess = mess.copy()
    for c_ in cols:
        if f'{c_}_{prefix}' not in mess.columns:
            mess[f'{c_}_{prefix}'] = np.nan
    m = mess[['CSN', 't_order'] + [f'{c_}_{prefix}' for c_ in cols]].copy()
    m.columns = ['CSN', 't_order'] + cols
    ma = m.rename(columns={c_: c_ + '_a' for c_ in cols}).rename(columns={'t_order': 'ta'})
    mb = m.rename(columns={c_: c_ + '_b' for c_ in cols}).rename(columns={'t_order': 'tb'})
    Q = P.merge(ma, on=['CSN', 'ta'], how='left').merge(mb, on=['CSN', 'tb'], how='left')
    Q['valid_a'] = Q.valid_a.fillna(0)
    Q['valid_b'] = Q.valid_b.fillna(0)
    Q['dT'] = Q.t_amp_b - Q.t_amp_a
    Q['dT_roh'] = Q.t_roh_b - Q.t_roh_a
    Q['dS'] = Q.steil_b - Q.steil_a
    Q['pol_sicher'] = (Q.pol_unsicher_a == 0) & (Q.pol_unsicher_b == 0)
    Q['pol_gleich'] = Q.pol_sicher & (Q.pol_a == Q.pol_b)
    Q['kollision'] = (Q.w0_b < Q.w1_a) & (Q.w0_a < Q.w1_b)
    h = hf.rename(columns={'t_order': 'ta', 'hf': 'hf_a'})
    Q = Q.merge(h, on=['CSN', 'ta'], how='left')
    h = hf.rename(columns={'t_order': 'tb', 'hf': 'hf_b'})
    Q = Q.merge(h, on=['CSN', 'tb'], how='left')
    Q['dhf'] = Q.hf_b - Q.hf_a
    Q = behandlung_kategorie(Q.rename(columns={'c_a': 'c_a', 'c_b': 'c_b'}), trt, cfg)
    return Q


def primaer_kette(Q, cfg=CFG, beh_ok=('kein_hinweis',)):
    """Zaehlkette und Primaermenge (Spezifikation 4.4). Rueckgabe (df, z, gruppen). beh_ok: erlaubte Behandlungskategorien.
    Primaer: kein_hinweis und Erstwert < 6,0. Die Kette ist sequenziell (jeder Schritt nur auf dem Rest des vorigen)."""
    z = {}
    z['paare_konsekutiv'] = int(len(Q))
    q = Q[~Q.ohne_ergebnis]
    z['nach_ausschluss_ohne_ergebniszeit'] = int(len(q))
    q = q[~q.zensiert]
    z['nach_ausschluss_zensiert'] = int(len(q))
    q = q[q.c_a.notna() & q.c_b.notna()]
    z['ecg_zeitlich_beide_enden'] = int(len(q))
    q = q[q.valid_a.eq(1) & q.valid_b.eq(1)]
    z['messung_gueltig_beide_enden'] = int(len(q))
    gruppen = {'pol_pp': int(((q.pol_a == 1) & (q.pol_b == 1) & q.pol_sicher).sum()), 'pol_mm': int(((q.pol_a == -1) & (q.pol_b == -1) & q.pol_sicher).sum()),
               'pol_gemischt': int(((q.pol_a != q.pol_b) & q.pol_sicher).sum()), 'pol_unsicher': int((~q.pol_sicher).sum())}
    q = q[q.pol_gleich]
    z['polaritaet_sicher_und_gleich'] = int(len(q))
    q = q[~q.kollision]
    z['ohne_fensterueberlappung'] = int(len(q))
    q = q[q.hf_a.notna() & q.hf_b.notna()]
    z['hf_beide_enden'] = int(len(q))
    z['davon_behandlung_dokumentiert'] = int((q.beh_kat == 'dokumentiert').sum())
    z['davon_behandlung_unklar'] = int((q.beh_kat == 'unklar').sum())
    q = q[q.beh_kat.isin(beh_ok)]
    z['behandlung_zugelassen'] = int(len(q))
    z['davon_erstwert_ab_6'] = int((q.ka >= cfg['primaer']['erst_k_max']).sum())
    q = q[q.ka < cfg['primaer']['erst_k_max']]
    z['primaerpaare'] = int(len(q))
    z['primaer_personen'] = int(q.MRN.nunique())
    return q.copy(), z, gruppen


def mit_hf_spline(q, cfg=CFG):
    """Spaltendifferenzen des HF-Splines f(HF2) - f(HF1) als dB1..dBk."""
    B = spline_differenz(q.hf_a.to_numpy(), q.hf_b.to_numpy(), cfg)
    q = q.copy()
    for j in range(B.shape[1]):
        q[f'dB{j + 1}'] = B[:, j]
    return q


def hf_spalten(cfg=CFG):
    return [f'dB{j + 1}' for j in range(len(cfg['hf_spline']['knoten']) - 1)]


# =========================================================================== Daten: Splits, Laden, Entnahmen (DuckDB)
class CSNMenge(list):
    """Besuchsmenge mit Splitzuordnung; wird nur von alle_csns und split_csns erzeugt."""

    def __init__(self, it, zuordnung, test):
        super().__init__(it)
        self.zuordnung = zuordnung
        self.enthaelt_test = test


def split_csns(datadir, gruppe, freigabe):
    """CSN-Menge eines Splits ('train', 'val', 'test') oder aller ('alle'). Jeder Zugriff, auch auf Train und Val, verlangt eine Freigabe (Bindungspruefung)."""
    pflicht_freigabe(freigabe)
    if gruppe not in ('train', 'val', 'test', 'alle'):
        raise ValueError(gruppe)
    namen = ['train', 'val', 'test'] if gruppe == 'alle' else [gruppe]
    z = {}
    for sp in namen:
        for csn in pd.read_csv(Path(datadir) / f'split_random_{sp}.csv', header=None)[0].astype('int64'):
            z[int(csn)] = sp
    return CSNMenge(sorted(z), z, 'test' in namen)


def alle_csns(datadir, freigabe):
    return split_csns(datadir, 'alle', freigabe)


def lies_kopf(path):
    """WFDB-Kopf: (Signalzahl, fs, Abtastwerte, Startzeit UTC-Sekunden, Einheit)."""
    L = Path(path).read_text().splitlines()
    a = L[0].split()
    b = L[1].split()
    tt = a[4] + ' ' + a[5]
    d = dt.datetime.strptime(tt, '%H:%M:%S.%f %d/%m/%Y') if '.' in a[4] else dt.datetime.strptime(tt, '%H:%M:%S %d/%m/%Y')
    return int(a[1]), float(a[2]), int(a[3]), d.replace(tzinfo=dt.timezone.utc).timestamp(), b[2].split('/')[-1]


def verbinden(pfad):
    import duckdb
    Path(pfad).parent.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect(str(pfad))
    c.execute("SET memory_limit='700MB'; SET threads=2")
    return c


def laden(c, datadir, wavedir, menge, freigabe):
    """Privaten Bestand bauen: sp, visit (mit Split), labs (nur benoetigte Komponenten, mit Einheit), orders (Medikation und Dialyse), seg (Ableitung-II-Koepfe mit Einheit),
    hrmon (Monitor-HF), meta (Freigabe-Hash). Verlangt eine Freigabe und eine von split_csns erzeugte Menge."""
    pflicht_freigabe(freigabe)
    if not isinstance(menge, CSNMenge):
        raise PermissionError('laden verlangt eine von split_csns erzeugte Besuchsmenge')
    datadir, wavedir = Path(datadir), Path(wavedir)
    c.execute('CREATE OR REPLACE TABLE meta(freigabe_hash VARCHAR, test_geladen BOOLEAN, vertrag VARCHAR)')
    c.execute('INSERT INTO meta VALUES (?,?,?)', [freigabe.hash, bool(menge.enthaelt_test), json.dumps(VERTRAG, sort_keys=True)])
    c.execute('CREATE OR REPLACE TABLE sp(CSN BIGINT, split VARCHAR)')
    c.register('csn_df', pd.DataFrame({'CSN': list(menge), 'split': [menge.zuordnung[x] for x in menge]}))
    c.execute('INSERT INTO sp SELECT CSN, split FROM csn_df')
    allv = "all_varchar=true,strict_mode=false,null_padding=true,ignore_errors=true"
    c.execute(f"""CREATE OR REPLACE TABLE visit AS SELECT CAST(v.CSN AS BIGINT) CSN, CAST(v.MRN AS BIGINT) MRN, sp.split, try_cast(v.Age AS DOUBLE) Age, v.Gender,
        epoch_us(try_cast(v.Roomed_time AS TIMESTAMP))/1e6 r, epoch_us(try_cast(v.Departure_time AS TIMESTAMP))/1e6 d
        FROM read_csv('{(datadir / 'visits.csv').as_posix()}', header=true, {allv}) v JOIN sp ON sp.CSN = try_cast(v.CSN AS BIGINT)""")
    namen = ",".join("'" + n + "'" for n in ['POTASSIUM', 'POC:POTASSIUM, ISTAT', 'POTASSIUM.WHOLE BLD', 'SODIUM', 'CO2', 'GLUCOSE', 'CALCIUM'])
    c.execute(f"""CREATE OR REPLACE TABLE labs AS SELECT try_cast(l.CSN AS BIGINT) CSN, epoch_us(try_cast(l.Order_time AS TIMESTAMP))/1e6 t_order,
        epoch_us(try_cast(l.Result_time AS TIMESTAMP))/1e6 t_result, l.Component_name cname, l.Component_value cval, l.Component_units unit
        FROM read_csv('{(datadir / 'labs.csv').as_posix()}', header=true, {allv}) l JOIN sp ON sp.CSN = try_cast(l.CSN AS BIGINT) WHERE l.Component_name IN ({namen})""")
    c.execute(f"""CREATE OR REPLACE TABLE orders AS SELECT try_cast(o.CSN AS BIGINT) CSN, o.Order_type otype, o.Procedure_name pname,
        epoch_us(try_cast(o.Order_time AS TIMESTAMP))/1e6 t_ord, epoch_us(try_cast(NULLIF(o.First_admin_time,'') AS TIMESTAMP))/1e6 t_admin
        FROM read_csv('{(datadir / 'orders.csv').as_posix()}', header=true, {allv}) o JOIN sp ON sp.CSN = try_cast(o.CSN AS BIGINT)
        WHERE o.Order_type = 'Medications' OR regexp_matches(upper(o.Procedure_name), 'DIALYZE|HEMODIALYSIS|CRRT|PERITONEAL DIALYSIS|CAPD|CCPD')""")
    c.execute(f"""CREATE OR REPLACE TABLE hrmon AS SELECT try_cast(n.CSN AS BIGINT) CSN, epoch_us(try_cast(n.Time AS TIMESTAMP))/1e6 t, try_cast(n.Value AS DOUBLE) hr
        FROM read_csv('{(datadir / 'numerics.csv').as_posix()}', header=true, {allv}) n JOIN sp ON sp.CSN = try_cast(n.CSN AS BIGINT)
        WHERE n.Source = 'Monitor' AND n.Measure = 'HR'""")
    rows = []
    for csn in menge:
        p = wavedir / str(csn)[-3:] / str(csn) / 'II'
        if not p.is_dir():
            continue
        for f in os.listdir(p):
            if f.endswith('.hea'):
                try:
                    ns, fs, n, t0, einheit = lies_kopf(p / f)
                except Exception:
                    continue
                rows.append((int(csn), int(f[:-4].rsplit('_', 1)[1]), fs, n, t0, t0 + n / fs, einheit))
    c.execute('CREATE OR REPLACE TABLE seg(CSN BIGINT, seg INT, fs DOUBLE, n BIGINT, t0 DOUBLE, t1 DOUBLE, einheit VARCHAR)')
    if rows:
        c.executemany('INSERT INTO seg VALUES (?,?,?,?,?,?,?)', rows)


def meta_pruefen(c, freigabe):
    pflicht_freigabe(freigabe)
    r = c.execute('SELECT freigabe_hash FROM meta').fetchall()
    if not r or r[0][0] != freigabe.hash:
        raise PermissionError('Bestand wurde nicht unter dieser Freigabe geladen')


def zeitvertrag_pruefen(c, min_besuche=20):
    """Zeitvertrag: Median(erster Monitor-HF-Zeitstempel minus erster Segmentbeginn) muss zwischen -60 und 300 s liegen (S5-Referenz rund 40 s). Rueckgabe Median."""
    r = c.execute("""SELECT median(h.t0 - s.t0), count(*) FROM (SELECT CSN, min(t) t0 FROM hrmon GROUP BY CSN) h JOIN (SELECT CSN, min(t0) t0 FROM seg GROUP BY CSN) s USING (CSN)""").fetchone()
    if r[1] is None or r[1] < min_besuche:
        raise ValueError(f'Zeitvertrag nicht pruefbar: nur {r[1]} Besuche mit Monitor-HF und Segment')
    if not (-60.0 <= r[0] <= 300.0):
        raise ValueError(f'Zeitvertrag verletzt: Median {r[0]:.0f} s ausserhalb -60 bis 300 s')
    return float(r[0])


def entnahmen_tabelle(c, freigabe, cfg=CFG):
    """Kaliumentnahmen je Besuch und Auftragszeit. Serum/Plasma (k_s) und andere Quellen (k_o: i-STAT, Vollblut) getrennt, nur Einheit mmol/L. Zensierte Werte (< oder >)
    tragen die Marke cens und werden nie als exakte Werte analysiert. Widerspruechliche Werte derselben Quelle und Auftragszeit entfallen (nk > 1). Panelwerte derselben
    Auftragszeit (Natrium, Bikarbonat nur mmol/L)."""
    meta_pruefen(c, freigabe)
    c.execute("""CREATE OR REPLACE TEMP TABLE kx AS SELECT CSN, t_order, min(t_result) t_result,
        avg(CASE WHEN cname='POTASSIUM' THEN k END) k_s, count(DISTINCT CASE WHEN cname='POTASSIUM' THEN k END) nk_s, coalesce(bool_or(CASE WHEN cname='POTASSIUM' THEN cens END), false) cens_s,
        avg(CASE WHEN cname<>'POTASSIUM' THEN k END) k_o, count(DISTINCT CASE WHEN cname<>'POTASSIUM' THEN k END) nk_o, coalesce(bool_or(CASE WHEN cname<>'POTASSIUM' THEN cens END), false) cens_o
        FROM (SELECT CSN, t_order, t_result, cname, try_cast(regexp_replace(cval,'^[<>]','') AS DOUBLE) k, (cval LIKE '>%' OR cval LIKE '<%') cens
              FROM labs WHERE cname IN ('POTASSIUM','POC:POTASSIUM, ISTAT','POTASSIUM.WHOLE BLD') AND unit = 'mmol/L') WHERE k IS NOT NULL GROUP BY CSN, t_order""")
    for nm, col in (('na', 'SODIUM'), ('hco3', 'CO2')):
        c.execute(f"CREATE OR REPLACE TEMP TABLE px_{nm} AS SELECT CSN, t_order, avg(try_cast(cval AS DOUBLE)) v FROM labs WHERE cname='{col}' AND unit='mmol/L' GROUP BY CSN, t_order")
    for nm, col in (('glc', 'GLUCOSE'), ('ca', 'CALCIUM')):
        c.execute(f"CREATE OR REPLACE TEMP TABLE px_{nm} AS SELECT CSN, t_order, avg(try_cast(cval AS DOUBLE)) v FROM labs WHERE cname='{col}' GROUP BY CSN, t_order")
    D = c.execute("""SELECT k.CSN, v.MRN, v.split, k.t_order, k.t_result, k.k_s, k.nk_s, k.cens_s, k.k_o, k.nk_o, k.cens_o, na.v na, hc.v hco3, gl.v glc, ca.v ca, v.Age age,
        (v.Gender='M')::INT sex FROM kx k JOIN visit v USING(CSN) LEFT JOIN px_na na ON na.CSN=k.CSN AND na.t_order=k.t_order LEFT JOIN px_hco3 hc ON hc.CSN=k.CSN AND hc.t_order=k.t_order
        LEFT JOIN px_glc gl ON gl.CSN=k.CSN AND gl.t_order=k.t_order LEFT JOIN px_ca ca ON ca.CSN=k.CSN AND ca.t_order=k.t_order""").df()
    for col in ('cens_s', 'cens_o'):
        D[col] = D[col].fillna(False).astype(bool)
    D['sex_m'] = D.sex
    return D


def einheiten_zaehlung(c):
    """Zaehlung der Kaliumzeilen mit abweichender Einheit (ausgeschlossen) und der Segmente mit abweichender Signaleinheit."""
    a = c.execute("SELECT count(*) FROM labs WHERE cname IN ('POTASSIUM','POC:POTASSIUM, ISTAT','POTASSIUM.WHOLE BLD') AND (unit IS NULL OR unit <> 'mmol/L')").fetchone()[0]
    b = c.execute("SELECT count(*) FROM seg WHERE lower(coalesce(einheit,'')) <> 'mv'").fetchone()[0]
    return {'kalium_zeilen_andere_einheit': int(a), 'segmente_andere_signaleinheit': int(b)}


def draws_serum(D):
    """Serumentnahmen: nur Serum/Plasma-Werte (Quelle bleibt auch bei gleichzeitigem i-STAT-Wert Serum); widerspruechliche Serumwerte entfallen."""
    d = D[D.k_s.notna() & (D.nk_s == 1)].copy()
    d['k'] = d.k_s
    d['cens'] = d.cens_s
    d['serum'] = True
    return d


def draws_alle(D):
    """Alle Quellen: Serum hat Vorrang; ohne Serum der eindeutige i-STAT-/Vollblutwert. Widerspruechliche Werte entfallen."""
    s = D.k_s.notna()
    k = np.where(s, np.where(D.nk_s == 1, D.k_s, np.nan), np.where(D.nk_o == 1, D.k_o, np.nan))
    cens = np.where(s, D.cens_s, D.cens_o)
    d = D.copy()
    d['k'] = k
    d['cens'] = cens.astype(bool)
    d['serum'] = s
    return d[d.k.notna()].copy()


def behandlungen_tabelle(c):
    t = c.execute("SELECT CSN, pname, t_admin, t_ord FROM orders").df()
    t['cls'] = t.pname.map(behandlung_klasse)
    return t.dropna(subset=['cls'])[['CSN', 'cls', 't_admin', 't_ord']].reset_index(drop=True)


def hf_tabelle(c, mitten, cfg=CFG):
    """Monitor-HF (Median +-5 min um die Blockmitte, mindestens 3 Werte). mitten: DataFrame (CSN, t_order, c). Rueckgabe (CSN, t_order, hf)."""
    fw = cfg['fenster']
    h = c.execute("SELECT CSN, t, hr FROM hrmon").df()
    g = {k: (v.t.to_numpy(), v.hr.to_numpy()) for k, v in h.groupby('CSN')}
    out = []
    for r in mitten.itertuples():
        v = g.get(r.CSN)
        if v is None or not np.isfinite(r.c):
            out.append(np.nan)
            continue
        w = v[1][(v[0] >= r.c - fw['hf_halbfenster_s']) & (v[0] <= r.c + fw['hf_halbfenster_s'])]
        out.append(float(np.median(w)) if len(w) >= fw['hf_min_werte'] else np.nan)
    m = mitten[['CSN', 't_order']].copy()
    m['hf'] = out
    return m


def h4_paare(Ds, cfg=CFG):
    """H4-Wiederholung: je Erstentnahme mit K >= 6,0 (nicht zensiert) die ERSTE spaetere Entnahme desselben Besuchs mit Abstand 0,25 bis 6 h (fuer alle gleich)."""
    c4 = cfg['h4']
    d = Ds[(Ds.k >= c4['k_erst_min']) & (~Ds.cens)].sort_values(['CSN', 't_order'])
    allv = Ds.sort_values(['CSN', 't_order'])
    g = {k: v for k, v in allv.groupby('CSN')}
    rows = []
    for r in d.itertuples():
        v = g[r.CSN]
        gap = (v.t_order - r.t_order) / 3600.0
        m = v[(gap >= c4['rep_min_h']) & (gap <= c4['rep_max_h'])]
        if len(m):
            f = m.iloc[0]
            rows.append({'CSN': r.CSN, 'MRN': r.MRN, 'split': r.split, 'ta': r.t_order, 'tb': f.t_order, 'ra': r.t_result, 'rb': f.t_result, 'ka': r.k, 'kb': f.k,
                         'zensiert': bool(r.cens or f.cens), 'gap_h': (f.t_order - r.t_order) / 3600.0})
    return pd.DataFrame(rows, columns=['CSN', 'MRN', 'split', 'ta', 'tb', 'ra', 'rb', 'ka', 'kb', 'zensiert', 'gap_h'])


def kandidaten(Dalle, Ps, Pa, H4p, seg, cfg=CFG):
    """Zu messende Entnahmen: Enden der Paare (Serum, alle Quellen) mit zeitlicher Abdeckung (>= 10 s Signal im D3-Fenster) an beiden Enden, Erstentnahmen der H4-Paare,
    Serumentnahmen mit K >= 6,5, und eine feste Zufallsstichprobe je Person (Serum) fuer die Querschnittsbeschreibung. Entnahmen ohne Ergebniszeit haben kein D3."""
    D = Dalle.copy()
    gr = [d3_grenzen(a, b, cfg) for a, b in zip(D.t_order, D.t_result)]
    D['lo'] = [g[0] if g else np.nan for g in gr]
    D['hi'] = [g[1] if g else np.nan for g in gr]
    sg_ = {k: v for k, v in seg[seg.einheit.str.lower() == 'mv'].groupby('CSN')}

    def abdeckung(r):
        s = sg_.get(r.CSN)
        if s is None or not np.isfinite(r.lo):
            return 0.0
        ov = np.minimum(s.t1.to_numpy(), r.hi) - np.maximum(s.t0.to_numpy(), r.lo)
        return float(max(0.0, ov.max())) if len(ov) else 0.0

    D['zeitl'] = [abdeckung(r) >= 10.0 for r in D.itertuples()]
    z = D.set_index(['CSN', 't_order']).zeitl.to_dict()
    ende = set()
    for P in (Ps, Pa):
        for r in P.itertuples():
            if z.get((r.CSN, r.ta), False) and z.get((r.CSN, r.tb), False):
                ende.add((r.CSN, r.ta))
                ende.add((r.CSN, r.tb))
    for r in H4p.itertuples():
        if z.get((r.CSN, r.ta), False):
            ende.add((r.CSN, r.ta))
    D['kand_paar'] = [(r.CSN, r.t_order) in ende for r in D.itertuples()]
    D['kand_h3'] = D.zeitl & D.serum & (D.k >= cfg['h3']['k_min']) & (~D.cens)
    rng = np.random.default_rng(cfg['seeds']['pksample'])
    ein = D[D.zeitl & D.serum & (~D.cens)].sample(frac=1.0, random_state=int(rng.integers(0, 2 ** 31))).drop_duplicates('MRN')
    ein = set(zip(ein.CSN, ein.t_order))
    D['kand_pk'] = [(r.CSN, r.t_order) in ein for r in D.itertuples()]
    return D[D.kand_paar | D.kand_h3 | D.kand_pk].reset_index(drop=True)


def segs_von(seg):
    d = {}
    for r in seg.itertuples():
        if str(r.einheit).lower() == 'mv':
            d.setdefault(int(r.CSN), []).append((int(r.seg), float(r.fs), int(r.n), float(r.t0), float(r.t1)))
    return d


def extrahieren(c, wavedir, freigabe, nw=1, cfg=CFG):
    """Entnahmen messen und in Tabelle mess_v2 des privaten Bestands schreiben (wiederaufnehmbar; keine Loeschbefehle)."""
    meta_pruefen(c, freigabe)
    D = entnahmen_tabelle(c, freigabe, cfg)
    Ds, Da = draws_serum(D), draws_alle(D)
    seg = c.execute('SELECT * FROM seg').df()
    K = kandidaten(Da, paare_bauen(Ds, cfg), paare_bauen(Da, cfg), h4_paare(Ds, cfg), seg, cfg)
    c.execute('CREATE TABLE IF NOT EXISTS mess_done(CSN BIGINT, t_order DOUBLE, pipeline VARCHAR)')
    fertig = set(map(tuple, c.execute("SELECT CSN, t_order FROM mess_done WHERE pipeline=?", [PIPELINE]).fetchall()))
    sg_ = segs_von(seg)
    arbeit = [(int(r.CSN), float(r.t_order), float(r.t_result) if r.t_result == r.t_result else float('nan'), sg_.get(int(r.CSN), []),
               {'rep': True, 'shift': bool(r.kand_paar), 'cfg': cfg, 'wave': str(wavedir)})
              for r in K.itertuples() if (r.CSN, r.t_order) not in fertig]
    res = []
    if nw > 1:
        from multiprocessing import Pool
        with Pool(nw) as p:
            for o in p.imap(messe_entnahme, arbeit, chunksize=4):
                res.append(o)
    else:
        res = [messe_entnahme(a) for a in arbeit]
    if res:
        M = pd.DataFrame(res)
        c.register('m_df', M)
        if c.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='mess_v2'").fetchone()[0]:
            cols = [r[0] for r in c.execute("DESCRIBE mess_v2").fetchall()]
            for col in M.columns:
                if col not in cols:
                    c.execute(f'ALTER TABLE mess_v2 ADD COLUMN "{col}" DOUBLE')
            c.execute('INSERT INTO mess_v2 BY NAME SELECT * FROM m_df')
        else:
            c.execute('CREATE TABLE mess_v2 AS SELECT * FROM m_df')
        c.executemany('INSERT INTO mess_done VALUES (?,?,?)', [(int(m.CSN), float(m.t_order), PIPELINE) for m in M.itertuples()])
    return {'kandidaten': int(len(K)), 'neu_gemessen': len(res)}


def tabellen_bauen(c, freigabe, cfg=CFG):
    """Aus dem privaten Bestand: Paartabellen (Serum D3, Serum D1, alle Quellen D3, zeitversetzt), Entnahmetabelle mit Messwerten, H4-Tabelle."""
    meta_pruefen(c, freigabe)
    D = entnahmen_tabelle(c, freigabe, cfg)
    trt = behandlungen_tabelle(c)
    M = c.execute('SELECT * FROM mess_v2').df()
    Ds, Da = draws_serum(D), draws_alle(D)
    Ps, Pa = paare_bauen(Ds, cfg), paare_bauen(Da, cfg)

    def hf_von(suffix):
        col = f'c_{suffix}'
        if col not in M:
            return pd.DataFrame({'CSN': [], 't_order': [], 'hf': []})
        return hf_tabelle(c, M[['CSN', 't_order', col]].rename(columns={col: 'c'}), cfg)

    hf3, hf1, hfs = hf_von('d3'), hf_von('d1'), hf_von('sh')
    T = {'Q': messwerte_anfuegen(Ps, M, 'd3', hf3, trt, cfg), 'Qd1': messwerte_anfuegen(Ps, M, 'd1', hf1, trt, cfg),
         'Qa': messwerte_anfuegen(Pa, M, 'd3', hf3, trt, cfg), 'Qsh': messwerte_anfuegen(Ps, M, 'sh', hfs, trt, cfg)}
    Dm = Ds.merge(M, on=['CSN', 't_order'], how='inner').merge(hf3, on=['CSN', 't_order'], how='left')
    for v in ('valid', 'pol', 'pol_unsicher', 't_amp', 't_roh', 'steil', 'qrs_ms', 'c', 'reason'):
        Dm[v] = Dm[f'{v}_d3'] if f'{v}_d3' in Dm else np.nan
    for v in ('valid_rep', 'pol_rep', 'pol_unsicher_rep', 't_amp_rep'):
        if v not in Dm:
            Dm[v] = np.nan
    Dm['valid'] = Dm.valid.fillna(0)
    Dm['valid_rep'] = Dm.valid_rep.fillna(0)
    T['Dm'] = Dm
    H = h4_paare(Ds, cfg)
    if len(H):
        m = M.rename(columns={'t_order': 'ta'})
        H = H.merge(m[['CSN', 'ta'] + [x for x in ('valid_d3', 'pol_unsicher_d3', 't_amp_d3', 'c_d3') if x in m]], on=['CSN', 'ta'], how='left')
        H = H.rename(columns={'valid_d3': 'valid_a', 'pol_unsicher_d3': 'pol_unsicher_a', 't_amp_d3': 't_amp_a', 'c_d3': 'c_a'})
        for v in ('valid_a', 'pol_unsicher_a', 't_amp_a', 'c_a'):
            if v not in H:
                H[v] = np.nan
        H['valid_a'] = H.valid_a.fillna(0)
        H['c_b'] = np.nan
        H['ohne_ergebnis'] = H.ra.isna() | H.rb.isna()
        H = behandlung_kategorie(H, trt, cfg)
    T['H4'] = H
    T['einheiten'] = einheiten_zaehlung(c)
    return T


# =========================================================================== Analyse
def praezisionsplanung(sd_dk, sd_res, n, deff=1.0):
    """Praezision einer Steigung: SE = sd_res / (sd_dk * sqrt(n / deff)); Halbbreite 95 % = 1,96 SE; kleinste bei 80 % Power nachweisbare Steigung = 2,8 SE.
    sd_dk = Streuung der Kaliumaenderung, sd_res = Residualstreuung der T-Aenderung, deff = Designeffekt der Personencluster."""
    se = sd_res / (sd_dk * math.sqrt(n / deff))
    return {'se': se, 'halbbreite95': 1.96 * se, 'mdb80': 2.8 * se}


def messfehler_szenarien(est, var_dk, cfg=CFG):
    """Szenarien fuer Messfehler im Kalium (klassisch, nicht differenziell): Zuverlaessigkeit der Differenz lambda = 1 - 2 sd^2 / Var(dK); korrigierte Steigung beta/lambda.
    Rein deterministische Skalierung der Punktschaetzung und Grenzen, keine Schaetzung des Messfehlers."""
    if not est.get('beurteilbar'):
        return {'beurteilbar': False}
    out = {'beurteilbar': True, 'szenarien': []}
    for sd in cfg['messfehler']['sd_k']:
        lam = 1.0 - 2.0 * sd ** 2 / var_dk
        if lam <= 0.2:
            out['szenarien'].append({'sd_k': sd, 'lambda': lam, 'beta': None})
            continue
        out['szenarien'].append({'sd_k': sd, 'lambda': lam, 'beta': est['beta'] / lam, 'ci_lo': est['ci_lo'] / lam, 'ci_hi': est['ci_hi'] / lam})
    return out


def h1_modelle(prim, cfg=CFG, y='dT', seed=None):
    """HF-adjustiert (Spline) und unadjustiert auf IDENTISCHEN Faellen (freier Achsenabschnitt)."""
    q = mit_hf_spline(prim, cfg).dropna(subset=[y, 'dk'] + hf_spalten(cfg))
    return q, schaetze(q, y, 'dk', hf_spalten(cfg), cfg, seed=seed), schaetze(q, y, 'dk', [], cfg, seed=seed)


def ausgangsbereich(q, cfg=CFG):
    kov = hf_spalten(cfg)
    bereiche = {'k1_bis_3.5': q.ka <= 3.5, 'k1_3.5_bis_5.0': (q.ka > 3.5) & (q.ka <= 5.0), 'k1_ueber_5.0': q.ka > 5.0}
    return {k: schaetze(q[m], 'dT', 'dk', kov, cfg, B=max(200, cfg['bootstrap_n'] // 4)) for k, m in bereiche.items()}


def timing_beschreibung(Q, cfg=CFG):
    """Abstand des ECG-Blocks zur Auftrags- und Ergebniszeit (Minuten) an beiden Enden der Paare mit ECG; Anteil vor Auftrag und nach Ergebnis."""
    out = {}
    for end in ('a', 'b'):
        c_, t_, r_ = Q[f'c_{end}'], Q['t' + end], Q['r' + end]
        ok = c_.notna() & r_.notna()
        da, dr = ((c_ - t_) / 60.0)[ok], ((c_ - r_) / 60.0)[ok]
        out[f'ende_{end}'] = {'n': int(ok.sum()), 'abstand_auftrag_min_q05_25_50_75_95': [float(x) for x in np.nanpercentile(da, [5, 25, 50, 75, 95])] if ok.sum() >= 10 else None,
                              'abstand_ergebnis_min_q05_25_50_75_95': [float(x) for x in np.nanpercentile(dr, [5, 25, 50, 75, 95])] if ok.sum() >= 10 else None,
                              'anteil_vor_auftrag': float((da < 0).mean()) if ok.sum() >= 10 else None, 'anteil_nach_ergebnis': float((dr > 0).mean()) if ok.sum() >= 10 else None}
    return out


def klinisch_h1(est, wh, prim, cfg=CFG):
    """Klinische Uebersetzung (Spezifikation 2): mm je mmol/L bei 10 mm/mV; Wiederholungsgrenze RC (95 % der Differenzen zweier Messungen derselben Entnahme) in mV und mm;
    individuelle T-Aenderung je Stufe der Kaliumaenderung gemessen an RC. Keine individuelle Nachweisgrenze, keine Aussage ueber visuelle Erkennbarkeit."""
    R = {'beurteilbar': bool(est.get('beurteilbar'))}
    if not est.get('beurteilbar'):
        return R
    f = cfg['klinisch']['mm_je_mv']
    R.update({'mm_je_mmol': est['beta'] * f, 'mm_je_mmol_ci': [est['ci_lo'] * f, est['ci_hi'] * f], 'mm_bei_dk_1': est['beta'] * f, 'mm_bei_dk_2': 2 * est['beta'] * f})
    rc = wh.get('rc') if wh and wh.get('beurteilbar') else None
    R['wiederholungsgrenze_mv'] = rc
    R['wiederholungsgrenze_mm'] = None if rc is None else rc * f
    R['erwartete_aenderung_je_mmol_im_verhaeltnis_zur_wiederholungsgrenze'] = None if not rc else est['beta'] / rc
    e = cfg['klinisch']['dk_strata']
    d = prim.dropna(subset=['dT', 'dk'])
    st = {'dk_ab_-1_abwaerts': d.dk <= e[0], 'dk_-1_bis_-0.5': (d.dk > e[0]) & (d.dk <= e[1]), 'dk_betrag_unter_0.5': (d.dk > e[1]) & (d.dk < e[2]),
          'dk_0.5_bis_1': (d.dk >= e[2]) & (d.dk < e[3]), 'dk_ab_1_aufwaerts': d.dk >= e[3]}
    tab = {}
    for k, m in st.items():
        g = d[m]
        o = {'n': int(len(g))}
        if len(g) >= 10:
            o['dT_median_mv'] = float(g.dT.median())
            o['dT_median_mm'] = float(g.dT.median() * f)
            o['dT_iqr_mv'] = [float(g.dT.quantile(.25)), float(g.dT.quantile(.75))]
            if rc:
                o['anteil_betrag_ueber_wiederholungsgrenze'] = float((g.dT.abs() > rc).mean())
                o['anteil_gleichgerichtet_ueber_wiederholungsgrenze'] = float(((np.sign(g.dT) == np.sign(g.dk)) & (g.dT.abs() > rc)).mean())
        tab[k] = o
    R['individuell_je_kaliumstufe'] = tab
    return R


def pk_beschreibung(D, cfg=CFG):
    """Beschreibung (kein Tor): Querschnittssteigung T-Amplitude je Kalium zwischen Personen, eine feste Zufallsentnahme je Person (hoechstens 6.000), adjustiert Alter, Geschlecht, HF."""
    d = D[(D.valid == 1) & (D.pol_unsicher == 0) & (~D.cens)].dropna(subset=['t_amp', 'k', 'hf', 'age', 'sex_m']).copy()
    d = d.sample(frac=1.0, random_state=int(np.random.default_rng(cfg['seeds']['pksample']).integers(0, 2 ** 31))).drop_duplicates('MRN').iloc[:6000]
    est = schaetze(d, 't_amp', 'k', ['age', 'sex_m', 'hf'], cfg, seed=cfg['seeds']['pk'])
    est['hinweis'] = 'beschreibend, zwischen Personen; kein Tor und kein Beleg fuer den Personeneffekt'
    return est


def wh_beschreibung(D):
    """Beschreibung (kein Tor): zwei getrennte 60-s-Fenster derselben Entnahme, gleiche sichere Polaritaet; Wiederholungsgrenze RC. Kein Beleg fuer Messrichtigkeit."""
    d = D[(D.valid == 1) & (D.valid_rep == 1) & (D.pol_unsicher == 0) & (D.pol_unsicher_rep == 0) & (D.pol == D.pol_rep)]
    e = icc_zweifenster(d.t_amp, d.t_amp_rep)
    e['hinweis'] = 'beschreibend; Wiederholbarkeit beweist keine Messrichtigkeit'
    return e


def h3_auswertung(D, cfg=CFG):
    """H3 (N3-M): QRS-Dauer gegen Bikarbonat bei K >= 6,5; nur Serum, nicht zensiert; eine (die erste) Entnahme je Person; adjustiert Kalium und Alter.
    Niedriges Bikarbonat ist kein Beweis fuer eine Azidose."""
    c3 = cfg['h3']
    d = D[(D.serum) & (~D.cens) & (D.k >= c3['k_min']) & (D.valid == 1)].dropna(subset=['qrs_ms', 'hco3', 'age']).sort_values('t_order').drop_duplicates('MRN').copy()
    R = {'n': int(len(d))}
    if len(d) < c3['n_min']:
        R['modell'] = {'beurteilbar': False, 'n': int(len(d))}
        R['kategorie'] = 'nicht_beurteilbar'
        return R
    R['modell'] = schaetze(d, 'qrs_ms', 'hco3', ['k', 'age'], cfg, seed=cfg['seeds']['bootstrap'] + 2)
    R['kategorie'] = kategorie_intervall(R['modell'], c3['grenze'], -1)
    q = d['hco3'].quantile([1 / 3, 2 / 3]).to_numpy()
    t = pd.cut(d.hco3, [-np.inf, q[0], q[1], np.inf], labels=['unteres', 'mittleres', 'oberes'])
    R['qrs_mittel_nach_bikarbonat_terzil'] = {str(k): {'n': int(v.size), 'mittel': float(v.mean())} for k, v in d.groupby(t, observed=True).qrs_ms}
    R['hinweis'] = 'Bikarbonat ist ein Surrogat; Azidose nicht belegt (Kompensation, Mischstoerungen); nur Gruppenmittel'
    return R


def h4_auswertung(H, cfg=CFG):
    """H4 (deskriptiv): persistierend erhoehtes (PE) gegen rasch normalisiertes (RN) Kalium ohne dokumentierte Behandlung. Erstwert >= 6,0, ERSTE spaetere Entnahme im
    Abstand 0,25 bis 6 h (fuer alle gleich). PE: Wiederholung >= 5,5; RN: Wiederholung < 5,0 und Abfall >= 1,0; unklar: dazwischen oder Behandlung unklar;
    dokumentierte Behandlung separat. Kein Anspruch auf Pseudohyperkaliaemie."""
    c4 = cfg['h4']
    if not len(H):
        return {'kategorie': 'nicht_beurteilbar', 'n_erstwerte': 0}
    q = H[(~H.zensiert) & (~H.ohne_ergebnis) & (H.valid_a == 1) & (H.pol_unsicher_a == 0)].copy()
    pe = (q.beh_kat == 'kein_hinweis') & (q.kb >= c4['pe_min'])
    rn = (q.beh_kat == 'kein_hinweis') & (q.kb < c4['rn_max']) & ((q.ka - q.kb) >= c4['rn_abfall'])
    dok = q.beh_kat == 'dokumentiert'
    q['gruppe'] = np.where(dok, 'dokumentiert', np.where(pe, 'PE', np.where(rn, 'RN', 'unklar')))
    R = {'n_erstwerte_mit_ecg': int(len(q)), 'pe': int(pe.sum()), 'rn': int(rn.sum()), 'unklar': int((q.gruppe == 'unklar').sum()), 'dokumentiert': int(dok.sum()),
         'rep_fenster_h': [c4['rep_min_h'], c4['rep_max_h']]}
    g = q[q.gruppe.isin(['PE', 'RN'])].copy()
    R['median_t_mv_je_gruppe'] = {k: (float(v.median()) if len(v) >= 10 else None) for k, v in q.groupby('gruppe').t_amp_a}
    if pe.sum() < c4['n_min_gruppe'] or rn.sum() < c4['n_min_gruppe']:
        R['kategorie'] = 'nicht_beurteilbar'
        return R
    g['is_pe'] = (g.gruppe == 'PE').astype(int)
    f = lambda t: np.array([auc_mw(t.loc[t.is_pe == 1, 't_amp_a'], t.loc[t.is_pe == 0, 't_amp_a'])])
    a = float(f(g)[0])
    lo, hi, _ = ci_perz(cluster_bootstrap(g, 'MRN', f, cfg['bootstrap_n'], cfg['seeds']['auc']))
    R.update({'auc': a, 'auc_ci': [lo, hi], 'auc_intervall_ueber_0.5': bool(lo > 0.5), 'auc_marke_erreicht': bool(a >= c4['auc_marke'])})
    # Erstkalium und Abstand beruecksichtigen: Residuen der T-Amplitude nach Regression auf K1 und Abstand (gepoolt), dann AUC der Residuen
    X = np.column_stack([np.ones(len(g)), g.ka.to_numpy(), g.gap_h.to_numpy()])
    g['t_res'] = g.t_amp_a.to_numpy() - X @ ols(X, g.t_amp_a.to_numpy())
    fr = lambda t: np.array([auc_mw(t.loc[t.is_pe == 1, 't_res'], t.loc[t.is_pe == 0, 't_res'])])
    lo2, hi2, _ = ci_perz(cluster_bootstrap(g, 'MRN', fr, cfg['bootstrap_n'], cfg['seeds']['auc'] + 2))
    R['auc_adjustiert_erstkalium_abstand'] = {'wert': float(fr(g)[0]), 'ci': [lo2, hi2]}
    # Beschreibende Marke: T-Amplitude >= 0,30 mV (3 mm); keine Entscheidungsgrenze
    thr = c4['schwelle_t_mv']
    fs_ = lambda t: np.array([(t.loc[t.is_pe == 1, 't_amp_a'] >= thr).mean(), (t.loc[t.is_pe == 0, 't_amp_a'] < thr).mean()])
    pt = fs_(g)
    bs = cluster_bootstrap(g, 'MRN', fs_, cfg['bootstrap_n'], cfg['seeds']['auc'] + 1)
    R['marke_t_mv'], R['marke_t_mm'] = thr, mm(thr, cfg)
    R['anteil_pe_ueber_marke'] = {'wert': float(pt[0]), 'ci': [float(np.nanpercentile(bs[:, 0], 2.5)), float(np.nanpercentile(bs[:, 0], 97.5))]}
    R['anteil_rn_unter_marke'] = {'wert': float(pt[1]), 'ci': [float(np.nanpercentile(bs[:, 1], 2.5)), float(np.nanpercentile(bs[:, 1], 97.5))]}
    # Unklare nach Folgewert zugeordnet (Sensitivitaet)
    u = q[q.gruppe.isin(['PE', 'RN', 'unklar'])].copy()
    u['is_pe'] = (u.kb >= (c4['pe_min'] + c4['rn_max']) / 2).astype(int)
    if u.is_pe.nunique() == 2:
        R['auc_unklare_nach_folgewert_zugeordnet'] = auc_mw(u.loc[u.is_pe == 1, 't_amp_a'], u.loc[u.is_pe == 0, 't_amp_a'])
    R['kategorie'] = 'beschrieben'
    R['hinweis'] = ('beschreibend ohne Entscheidungsanspruch: Gruppen aus Verlauf und dokumentierter Behandlung; Zeit, Regression zur Mitte, Probenfehler und unerfasste '
                    'Behandlung koennen sie vermischen; kein Referenzstandard, keine Pseudohyperkaliaemie-Aussage, keine Praevalenzuebertragung')
    return R


def nk1_auswertung(prim, Qsh, beta_primaer, cfg=CFG):
    """NK1: zeitversetzte ECG-Fenster derselben Person (2 bis 6 h neben der Entnahme, ausserhalb D3; tatsaechlich gewaehlter Block und Fenster geprueft). Labor unveraendert.
    Gleiche Paare wie die Primaermenge, soweit beide versetzten Messungen gueltig, sicher, gleichpolig und ohne Fensterueberlappung sind. Keine Mindestzahl, aber Praezisionsangabe."""
    if Qsh is None or not len(Qsh):
        return {'n': 0, 'kategorie': 'nicht_beurteilbar'}
    keys = prim[['CSN', 'ta', 'tb']]
    q = Qsh.merge(keys, on=['CSN', 'ta', 'tb'])
    q = q[q.valid_a.eq(1) & q.valid_b.eq(1) & q.pol_gleich & ~q.kollision & q.hf_a.notna() & q.hf_b.notna()]
    if len(q) < cfg['nk']['n_min']:
        return {'n': int(len(q)), 'kategorie': 'nicht_beurteilbar'}
    q = mit_hf_spline(q, cfg)
    e = schaetze(q, 'dT', 'dk', hf_spalten(cfg), cfg, seed=cfg['seeds']['nk1'], B=max(200, cfg['bootstrap_n'] // 2))
    if not e.get('beurteilbar'):
        e['kategorie'] = 'nicht_beurteilbar_praezise'
        e['halbbreite'] = None
        return e
    e['halbbreite'] = (e['ci_hi'] - e['ci_lo']) / 2
    e['kategorie'] = kategorie_nullnaehe(e, cfg['nk']['grenze'])
    e['beta_primaer_gleiche_paare'] = schaetze(mit_hf_spline(prim.merge(q[['CSN', 'ta', 'tb']], on=['CSN', 'ta', 'tb']), cfg), 'dT', 'dk', hf_spalten(cfg), cfg,
                                              B=max(200, cfg['bootstrap_n'] // 4)).get('beta')
    return e


def nk2_permutationen(n_zeilen, mrn, n_perm, seed):
    """Feste Vollstichprobe: Personenbloecke in zufaelliger Reihenfolge, Zuordnung durch zyklische Verschiebung um mindestens die groesste Blockgroesse.
    Damit erhaelt keine Zeile die Daten der eigenen Person, und jede Permutation nutzt alle Zeilen. Rueckgabe Liste von Indexarrays oder None, wenn nicht moeglich."""
    rng = np.random.default_rng(seed)
    mrn = np.asarray(mrn)
    _, inv, cnt = np.unique(mrn, return_inverse=True, return_counts=True)
    gmax = int(cnt.max())
    if n_zeilen < 2 * gmax or len(cnt) < 2:
        return None
    gr = [np.where(inv == i)[0] for i in range(len(cnt))]
    out = []
    for _ in range(n_perm):
        reihenfolge = rng.permutation(len(gr))
        idx = np.concatenate([gr[i] for i in reihenfolge])
        shift = int(rng.integers(gmax, n_zeilen - gmax + 1))
        p = np.empty(n_zeilen, int)
        p[idx] = np.roll(idx, -shift)
        out.append(p)
    return out


def nk2_auswertung(prim, cfg=CFG):
    """NK2: (dT, HF-Splinedifferenz) einer Paarzeile wird einer Zeile einer anderen Person zugeordnet (feste Vollstichprobe, 500 Permutationen). Prueft die Rechenpipeline
    (Kalibrierung des Schaetzers), nicht die Physiologie."""
    q = mit_hf_spline(prim, cfg).dropna(subset=['dT', 'dk'] + hf_spalten(cfg)).reset_index(drop=True)
    perms = nk2_permutationen(len(q), q.MRN.to_numpy(), cfg['nk2_n'], cfg['seeds']['nk2']) if len(q) >= cfg['min_schaetzung']['paare'] else None
    if perms is None:
        return {'n': int(len(q)), 'kategorie': 'nicht_beurteilbar'}
    kov = hf_spalten(cfg)
    obs = float(steigung(q, 'dT', 'dk', kov))
    out = []
    for p in perms:
        t = q.copy()
        for col in ['dT'] + kov:
            t[col] = q[col].to_numpy()[p]
        out.append(float(steigung(t, 'dT', 'dk', kov)))
    out = np.array(out)
    est = {'beurteilbar': True, 'ci_lo': float(np.percentile(out, 2.5)), 'ci_hi': float(np.percentile(out, 97.5))}
    return {'n': int(len(q)), 'beta_beobachtet': obs, 'null_mittel': float(out.mean()), 'null_sd': float(out.std(ddof=1)), 'q025': est['ci_lo'], 'q975': est['ci_hi'],
            'p_perm': float(((np.abs(out) >= abs(obs)).sum() + 1) / (len(out) + 1)), 'kategorie': kategorie_nullnaehe(est, cfg['nk']['grenze']),
            'hinweis': 'prueft die Rechenpipeline (Kalibrierung), keine Physiologie und kein Falsifikationstest fuer H1'}


def na_kontrolle(prim, cfg=CFG):
    """Pipeline-Kontrolle Natrium (keine gesicherte Nullexposition): Steigung dT je 5 mmol/L Natrium, HF-adjustiert; bei |r(dNa, dK)| > 0,30 partiell auf dK."""
    q = mit_hf_spline(prim, cfg).dropna(subset=['dT', 'dna', 'dk'] + hf_spalten(cfg)).copy()
    if len(q) < cfg['min_schaetzung']['paare']:
        return {'n': int(len(q)), 'kategorie': 'nicht_beurteilbar'}
    q['dna5'] = q.dna / cfg['na']['schritt']
    kov = hf_spalten(cfg)
    roh = schaetze(q, 'dT', 'dna5', kov, cfg, seed=cfg['seeds']['bootstrap'] + 3)
    par = schaetze(q, 'dT', 'dna5', kov + ['dk'], cfg, seed=cfg['seeds']['bootstrap'] + 4)
    r = float(np.corrcoef(q.dna, q.dk)[0, 1])
    rein = abs(r) <= cfg['na']['korr_max']
    mass = roh if rein else par
    return {'n': int(len(q)), 'roh': roh, 'partiell_dk': par, 'korrelation_dna_dk': r, 'kontrolle_rein': bool(rein), 'massgebend': 'roh' if rein else 'partiell_dk',
            'kategorie': kategorie_nullnaehe(mass, cfg['na']['grenze']), 'hinweis': 'Pipeline-Kontrolle; Natrium ist an Kalium gekoppelt (Behandlung, Dialyse) und keine gesicherte Nullexposition'}


def stabilitaet_je_split(prim, cfg=CFG):
    """Ergaenzende Stabilitaetsbeschreibung: dieselbe HF-adjustierte Steigung je Split (kein Bestaetigungsanspruch)."""
    q = mit_hf_spline(prim, cfg).dropna(subset=['dT', 'dk'] + hf_spalten(cfg))
    return {str(s): schaetze(g, 'dT', 'dk', hf_spalten(cfg), cfg, B=max(200, cfg['bootstrap_n'] // 4)) for s, g in q.groupby('split')}


def h1_auswertung(Q, cfg=CFG, wh=None):
    """H1 (primaer): QRS-relative T-Amplitude, Paardifferenzen mit freiem Achsenabschnitt, Personen-Clusterbootstrap, HF als Spline-Differenz; Primaermenge = Behandlung kein_hinweis
    und Erstwert < 6,0. Unadjustiert und adjustiert auf identischen Faellen. Sensitivitaeten (Spezifikation 6.1). Immer Schaetzer und Intervalle; Urteile nur bei Schaetzbarkeit."""
    prim, z, gruppen = primaer_kette(Q, cfg)
    R = {'zaehlkette': zaehlkette_vergroebern(z), 'polaritaetsgruppen': gruppen, 'zaehlkette_roh': z}
    q, adj, unadj = h1_modelle(prim, cfg)
    R['primaer'], R['unadjustiert_gleiche_faelle'] = adj, unadj
    R['kategorie'] = kategorie_intervall(adj, cfg['h1']['grenze'])
    kov = hf_spalten(cfg)
    S = {}
    S['huber'] = schaetze_huber(q, 'dT', 'dk', kov, cfg)
    S['einfluss'] = einfluss_pruefung(q, 'dT', 'dk', kov, cfg)
    S['ausgangsbereich'] = ausgangsbereich(q, cfg)
    B4 = max(200, cfg['bootstrap_n'] // 4)
    S['nur_anstieg'] = schaetze(q[q.dk > 0], 'dT', 'dk', kov, cfg, B=B4)
    S['nur_abfall'] = schaetze(q[q.dk < 0], 'dT', 'dk', kov, cfg, B=B4)
    S['mit_k1'] = schaetze(q, 'dT', 'dk', kov + ['ka'], cfg, B=B4)
    S['mit_bikarbonat'] = schaetze(q, 'dT', 'dk', kov + ['dhco3'], cfg, B=B4)
    S['nur_abs_dhf_unter_10'] = schaetze(q[q.dhf.abs() < 10], 'dT', 'dk', kov, cfg, B=B4)
    S['eine_beobachtung_je_person'] = schaetze(q.sort_values('ta').drop_duplicates('MRN'), 'dT', 'dk', kov, cfg, B=B4)
    S['gemischtes_modell'] = gemischtes_modell(q, 'dT', 'dk', kov, adj.get('beta'))
    p2, z2, _g = primaer_kette(Q, cfg, beh_ok=('kein_hinweis', 'unklar'))
    S['ohne_dokumentierte_behandlung'] = {'primaerpaare': int(len(p2)), 'modell': h1_modelle(p2, cfg)[1]}
    cfg_e = copy.deepcopy(cfg)
    cfg_e['primaer']['erst_k_max'] = 1e9
    p3, _z3, _g = primaer_kette(Q, cfg_e)
    S['alle_erstwerte'] = {'primaerpaare': int(len(p3)), 'modell': h1_modelle(p3, cfg)[1]}
    # rohe vorzeichenbehaftete Ableitung-II-Amplitude
    S['roh_t_amplitude_primaerpaare'] = h1_modelle(prim, cfg, y='dT_roh')[1]
    qa = Q[(~Q.ohne_ergebnis) & (~Q.zensiert) & Q.valid_a.eq(1) & Q.valid_b.eq(1) & Q.hf_a.notna() & Q.hf_b.notna() & (Q.beh_kat == 'kein_hinweis') & (Q.ka < cfg['primaer']['erst_k_max'])]
    S['roh_t_amplitude_alle_gueltigen_paare_inkl_polaritaetswechsel'] = {'primaerpaare': int(len(qa)), 'modell': h1_modelle(qa, cfg, y='dT_roh')[1]}
    var_dk = float(q.dk.var()) if len(q) > 2 else float('nan')
    S['messfehler_kalium'] = messfehler_szenarien(adj, var_dk, cfg)
    R['sensitivitaeten'] = S
    # Behandlungsstratum dokumentiert (nur beschreibend)
    pd_, zd, _g = primaer_kette(Q, cfg, beh_ok=('dokumentiert',))
    R['behandlung_dokumentiert'] = {'paare': int(len(pd_)), 'modell': schaetze(mit_hf_spline(pd_, cfg), 'dT', 'dk', kov, cfg, B=B4) if len(pd_) >= 100 else 'nicht_geschaetzt_n_unter_100'}
    # H2
    q2 = q.dropna(subset=['dS'])
    R['h2_steilheit'] = schaetze(q2, 'dS', 'dk', kov, cfg, seed=cfg['seeds']['bootstrap'] + 1)
    R['h2_kategorie'] = kategorie_intervall(R['h2_steilheit'], cfg['h2']['grenze'])
    # Praezision beobachtet und Planung
    if adj.get('beurteilbar'):
        X = np.column_stack([np.ones(len(q)), q.dk.to_numpy()] + [q[k].to_numpy() for k in kov])
        res = q.dT.to_numpy() - X @ ols(X, q.dT.to_numpy())
        sd_res, sd_dk = float(res.std(ddof=X.shape[1])), float(q.dk.std())
        mpp = float(len(q) / max(1, q.MRN.nunique()))
        R['praezision'] = {'sd_dk': sd_dk, 'sd_residual_mv': sd_res, 'paare_je_person': mpp, 'se_boot': adj['se_boot'], 'halbbreite95': (adj['ci_hi'] - adj['ci_lo']) / 2,
                           'planung_formel': praezisionsplanung(sd_dk, sd_res, len(q), deff=mpp)}
    R['timing'] = timing_beschreibung(Q[Q.set_index(['CSN', 'ta', 'tb']).index.isin(list(zip(prim.CSN, prim.ta, prim.tb)))], cfg)
    R['klinisch'] = klinisch_h1(adj, wh, prim, cfg)
    R['_prim'] = prim
    return R


def d1_vergleich(prim3, Qd1, cfg=CFG):
    """D1 gegen D3 auf denselben Paaren: Paare, die unter D3 UND D1 in der Primaermenge sind; je Fenster die eigenen Messwerte."""
    if Qd1 is None:
        return {'beurteilbar': False}
    p1, z1, _g = primaer_kette(Qd1, cfg)
    k = p1[['CSN', 'ta', 'tb']].merge(prim3[['CSN', 'ta', 'tb']], on=['CSN', 'ta', 'tb'])
    a = prim3.merge(k, on=['CSN', 'ta', 'tb'])
    b = p1.merge(k, on=['CSN', 'ta', 'tb'])
    return {'beurteilbar': True, 'primaerpaare_d1_gesamt': int(len(p1)), 'gemeinsame_paare': int(len(k)), 'd3': h1_modelle(a, cfg)[1], 'd1': h1_modelle(b, cfg)[1]}


def alle_quellen(Qa, cfg=CFG):
    p, z, _g = primaer_kette(Qa, cfg)
    return {'primaerpaare': int(len(p)), 'modell': h1_modelle(p, cfg)[1]}


# =========================================================================== Deutung und Gesamtauswertung
ZEITSATZ = ('Die Abnahmezeit ist nicht dokumentiert (nur Auftrag und Ergebnis); der Befund gilt fuer das ECG naechst der Auftragszeit innerhalb des Fensters '
            'Auftrag minus 15 min bis Ergebnis plus 15 min.')


def deutung(R, cfg=CFG):
    """Deutungstabelle (Spezifikation 9). Kontrollen, die von der Nullnaehe abweichen ('von_null_verschieden'), machen H1 und H2 nicht deutbar."""
    S = []
    kon = R.get('kontrollen', {})
    auff = [k for k, v in kon.items() if isinstance(v, dict) and v.get('kategorie') == 'von_null_verschieden']
    h1 = R.get('h1', {})
    texte = {
        'positiv_ueber_grenze': 'Innerhalb der Person geht eine Kaliumaenderung mit einer gleichgerichteten Aenderung der QRS-relativen T-Amplitude einher; Effekte unterhalb von 0,010 mV je mmol/L (0,1 mm) sind ausgeschlossen.',
        'positiv_unter_grenze': 'Gleichgerichteter Zusammenhang nachweisbar; Effekte ab 0,010 mV je mmol/L sind ausgeschlossen.',
        'positiv_grenze_offen': 'Gleichgerichteter Zusammenhang nachweisbar; ob er die Grenze 0,010 mV je mmol/L erreicht, bleibt offen.',
        'grenze_ausgeschlossen': 'Kein Nachweis; Effekte ab 0,010 mV je mmol/L sind ausgeschlossen.',
        'unpraezise': 'Weder ein Zusammenhang noch ein Effekt ab 0,010 mV je mmol/L ausgeschlossen.',
        'negativ': 'Negativer Zusammenhang (entgegen der Erwartung); nach technischer Pruefung (Polaritaet, Kontrollen) interpretierbar.',
        'nicht_beurteilbar': 'Zu wenige Paare oder Personen fuer eine Schaetzung.'}
    if h1:
        if auff:
            S.append(('H1', 'nicht_deutbar', 'Kontrolle(n) ' + ', '.join(auff) + ' weichen von der Nullnaehe ab: Artefakt (Messkette, Selektion, Behandlung) vorrangig; Schaetzer und Intervalle werden berichtet, nicht gedeutet.'))
        else:
            S.append(('H1', h1.get('kategorie'), texte.get(h1.get('kategorie'), '') + ' ' + ZEITSATZ))
        if h1.get('klinisch', {}).get('beurteilbar'):
            k = h1['klinisch']
            rc = k.get('wiederholungsgrenze_mm')
            S.append(('H1 klinisch', 'uebersetzt', 'Entspricht %.2f mm je mmol/L bei Standardverstaerkung (10 mm/mV); %s Aussage ueber die Gruppe der Paare; keine individuelle Nachweisgrenze und keine Aussage ueber visuelle Erkennbarkeit.'
                      % (k['mm_je_mmol'], ('Wiederholungsgrenze zweier Messungen derselben Entnahme %.2f mm.' % rc) if rc is not None else 'Wiederholungsgrenze nicht verfuegbar.')))
        if 'h2_kategorie' in h1:
            S.append(('H2', 'nicht_deutbar' if auff else h1['h2_kategorie'], 'T-Steilheit (mV/s je mmol/L), Grenze 0,2; Kategorien wie H1.' + (' Kontrollen auffaellig.' if auff else '')))
    h3 = R.get('h3', {})
    if h3:
        S.append(('H3', h3.get('kategorie', 'nicht_beurteilbar'), 'N3-M, Gruppenmittel bei K >= 6,5 (Serum), Bikarbonat ist ein Surrogat und kein Beweis einer Azidose; Grenze 0,29 ms je mmol/L (Haelfte von BM-014).'))
    h4 = R.get('h4', {})
    if h4:
        S.append(('H4', h4.get('kategorie', 'nicht_beurteilbar'), 'Beschreibung: persistierend erhoehtes gegen rasch normalisiertes Kalium ohne dokumentierte Behandlung; Zeit, Regression zur Mitte, Probenfehler und unerfasste Behandlung koennen die Gruppen vermischen; kein Pseudohyperkaliaemie-Anspruch.'))
    return S


def gesamt(T, cfg=CFG):
    """Die eine Gesamtauswertung aus den Tabellen T (Q, Qd1, Qa, Qsh, Dm, H4). Es gibt keine Tore: Schaetzer, Intervalle und Kontrollen werden immer berichtet.
    Nur eine nicht schaetzbare Primaermenge (< 30 Paare oder < 10 Personen) liefert 'nicht_beurteilbar'."""
    R = {'version': cfg['version'], 'pipeline': PIPELINE}
    Dm = T['Dm']
    R['beschreibungen'] = {'pk_querschnitt': pk_beschreibung(Dm, cfg), 'wiederholbarkeit': wh_beschreibung(Dm), 'einheiten': T.get('einheiten')}
    R['h1'] = h1_auswertung(T['Q'], cfg, R['beschreibungen']['wiederholbarkeit'])
    prim = R['h1'].pop('_prim')
    R['h1']['sensitivitaeten']['d1_gleiche_paare'] = d1_vergleich(prim, T.get('Qd1'), cfg)
    R['h1']['sensitivitaeten']['alle_quellen'] = alle_quellen(T['Qa'], cfg) if T.get('Qa') is not None else {'beurteilbar': False}
    R['h1']['stabilitaet_je_split'] = stabilitaet_je_split(prim, cfg)
    R['kontrollen'] = {'nk1': nk1_auswertung(prim, T.get('Qsh'), R['h1']['primaer'].get('beta', 0.0), cfg), 'nk2': nk2_auswertung(prim, cfg), 'na_pipeline': na_kontrolle(prim, cfg)}
    R['h3'] = h3_auswertung(Dm, cfg)
    R['h4'] = h4_auswertung(T['H4'], cfg)
    R['deutung'] = deutung(R, cfg)
    return R


def ausgabe_maskieren(R):
    """Ausgabe: Zaehlketten vergroebert (Komplemente), dann Kleinzellenregel nach Zaehlfeldern."""
    R = copy.deepcopy(R)
    if 'zaehlkette_roh' in R.get('h1', {}):
        R['h1'].pop('zaehlkette_roh')
    return mask_tree(R)


# =========================================================================== Gesamtlauf
EINGABEN = ('labs.csv', 'orders.csv', 'visits.csv', 'numerics.csv', 'split_random_train.csv', 'split_random_val.csv', 'split_random_test.csv')


def eingaben_pruefen(datadir, dateien=EINGABEN):
    """Prueft die Eingabedateien (inklusive Testsplit, erst nach der Freigabe) gegen SHA256SUMS.txt des Datensatzes (eine Ebene ueber data/). Fehlt die Datei, wird dies gemeldet."""
    sums = Path(datadir).parent / 'SHA256SUMS.txt'
    if not sums.exists():
        return {'geprueft': False, 'grund': 'SHA256SUMS.txt fehlt'}
    soll = {}
    for line in sums.read_text(encoding='utf-8', errors='replace').splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            soll[parts[1].strip().lstrip('*')] = parts[0].lower()
    for f in dateien:
        key = f'data/{f}'
        if key not in soll:
            raise ValueError('nicht in SHA256SUMS: ' + key)
        if sha(Path(datadir) / f) != soll[key]:
            raise ValueError('Hash weicht ab: ' + key)
    return {'geprueft': True, 'dateien': len(dateien)}


def gesamtlauf(datadir=DATA, wavedir=WAVE, priv=PRIV, lock=None, hash_fn=None, cfg=CFG, nw=1, ausgabe=None):
    """Die eine Auswertung: Bindung pruefen -> alle Splits laden -> Zeitvertrag pruefen -> messen -> Tabellen -> gesamt -> maskierte Ausgabe."""
    global WAVE
    fr = pruefe_start(lock, hash_fn)
    WAVE = Path(wavedir)
    eingaben = eingaben_pruefen(datadir)
    menge = alle_csns(datadir, fr)
    c = verbinden(Path(priv) / 's3_gesamt.duckdb')
    laden(c, datadir, wavedir, menge, fr)
    zv = zeitvertrag_pruefen(c)
    info = extrahieren(c, wavedir, fr, nw, cfg)
    T = tabellen_bauen(c, fr, cfg)
    R = gesamt(T, cfg)
    R['lauf'] = {'zeitvertrag_median_s': zv, 'extraktion': info, 'vertrag': VERTRAG, 'eingaben': eingaben}
    out = ausgabe_maskieren(R)
    ap = Path(ausgabe or (Path(priv) / 'auswertung_v2'))
    ap.mkdir(parents=True, exist_ok=True)
    (ap / 'gesamt_maskiert.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    c.close()
    return out


# =========================================================================== Kaliumblinde Referenzannotation (optional)
def annotation_stichprobe(Dm, n=None, seed=None, cfg=CFG):
    """Zufaellige Bloecke fuer die kaliumblinde Referenzannotation: eine Entnahme je Person, unabhaengig von Kalium, Behandlung und Messergebnis gezogen.
    Rueckgabe: (Annotationsliste ohne Kalium und ohne Pipelinewerte, Schluesseltabelle mit CSN und Zeit). Die Annotations-ID ist zufaellig."""
    n = n or cfg['annotation']['n']
    d = Dm[Dm.c.notna()].drop_duplicates('MRN')
    rng = np.random.default_rng(seed if seed is not None else cfg['seeds']['annotation'])
    idx = rng.permutation(len(d))[:n]
    s = d.iloc[idx].copy()
    s['annot_id'] = [f'A{int(x):06d}' for x in rng.permutation(10 ** 6)[:len(s)]]
    return s[['annot_id']].reset_index(drop=True), s[['annot_id', 'CSN', 'c']].reset_index(drop=True)


def annotation_auswertung(anwender, pipeline):
    """Vergleich der Anwender-T-Amplitude (mV) mit der Pipeline: Bias, Streuung der Differenz, Grenzen der Uebereinstimmung, ICC(1,1), Spearman."""
    from scipy.stats import spearmanr
    m = anwender.merge(pipeline, on='annot_id')
    e = icc_zweifenster(m.t_anwender, m.t_pipeline)
    if e.get('beurteilbar'):
        e['spearman'] = float(spearmanr(m.t_anwender, m.t_pipeline).correlation)
        e['anteil_innerhalb_0.05_mv'] = float((np.abs(m.t_anwender - m.t_pipeline) <= 0.05).mean())
    return e


def annotation_bild(x, ts, c, pfad):
    """Bild fuer die kaliumblinde Annotation: 10-s-Streifen um die Blockmitte c und Medianschlag, Raster 25 mm/s und 10 mm/mV. Keine Kalium-, Behandlungs- oder Pipelinewerte."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    i0 = max(0, int(round((c - 5 - ts) * FS)))
    i1 = min(len(x), int(round((c + 5 - ts) * FS)))
    seg = np.asarray(x[i0:i1], float)
    t = np.arange(len(seg)) / FS
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, seg, lw=0.8, color='black')
    ax.set_xlabel('s (25 mm/s)')
    ax.set_ylabel('mV (10 mm/mV)')
    ax.set_xticks(np.arange(0, t[-1] + 0.2, 0.2), minor=True)
    ax.set_yticks(np.arange(np.floor(seg.min() * 10) / 10, seg.max() + 0.1, 0.1), minor=True)
    ax.grid(which='minor', color='#e8b4b4', lw=0.4)
    ax.grid(which='major', color='#c97878', lw=0.6)
    fig.tight_layout()
    fig.savefig(pfad, dpi=110)
    plt.close(fig)


def annotation_vorbereiten(datadir=DATA, wavedir=WAVE, priv=PRIV, lock=None, hash_fn=None, cfg=CFG, n=None, ausgabe=None):
    """Optional, vor dem Gesamtlauf (nach der Bindung): kaliumblinde Referenzannotation. Nur Train und Val (der Testsplit bleibt zu), zufaellige Bloecke aus Entnahmen mit Block,
    unabhaengig von Kalium. Schreibt Bilder und eine Liste ohne Kalium und ohne Pipelinewerte; die Schluesseltabelle bleibt privat."""
    global WAVE
    fr = pruefe_start(lock, hash_fn)
    WAVE = Path(wavedir)
    tr, va = split_csns(datadir, 'train', fr), split_csns(datadir, 'val', fr)
    menge = CSNMenge(sorted(set(tr) | set(va)), {**tr.zuordnung, **va.zuordnung}, False)
    c = verbinden(Path(priv) / 's3_annotation.duckdb')
    laden(c, datadir, wavedir, menge, fr)
    extrahieren(c, wavedir, fr, 1, cfg)
    Dm = tabellen_bauen(c, fr, cfg)['Dm']
    liste, schluessel = annotation_stichprobe(Dm, n, None, cfg)
    ap = Path(ausgabe or (Path(priv) / 'annotation'))
    ap.mkdir(parents=True, exist_ok=True)
    segs = segs_von(c.execute('SELECT * FROM seg').df())
    bilder = []
    for r in schluessel.itertuples():
        seg = [s for s in segs.get(int(r.CSN), []) if s[3] <= r.c <= s[4]]
        if not seg:
            bilder.append('')
            continue
        rd = lies_roh(int(r.CSN), seg[0], r.c - 5, r.c + 5, wavedir)
        p = ap / f'{r.annot_id}.png'
        annotation_bild(rd[1], rd[0], r.c, p)
        bilder.append(p.name)
    liste['bild'] = bilder
    liste.to_csv(ap / 'annotation_liste.csv', index=False)
    schluessel.to_csv(Path(priv) / 'annotation_schluessel_privat.csv', index=False)
    c.close()
    return liste


# =========================================================================== Kunstdaten
def kunst_ecg(dauer_s, hr=70.0, t_amp=0.3, pol=1, rng=None, rausch=0.01, fs=FS, qrs_breite=1.0):
    """Synthetisches EKG (Gauss-Wellen), lokal je Schlag gerechnet; T-Amplitude vorgegeben; pol = -1 dreht das gesamte Signal."""
    rng = rng or np.random.default_rng(0)
    n = int(dauer_s * fs)
    x = np.zeros(n)
    rr = 60.0 / hr
    sq = np.sqrt(rr)
    wellen = ((0.12, -0.16, 0.025), (-0.10, -0.028, 0.010), (1.0, 0.0, 0.011 * qrs_breite), (-0.18, 0.026, 0.010), (t_amp, 0.30 * sq, 0.045))
    b = 0.5
    while b < dauer_s - 1.0:
        r = b + rng.normal(0, 0.005)
        i0, i1 = max(0, int((r - 0.5) * fs)), min(n, int((r + 0.9) * fs))
        t = np.arange(i0, i1) / fs
        for amp, off, sd in wellen:
            x[i0:i1] += amp * np.exp(-0.5 * ((t - (r + off)) / sd) ** 2)
        b += rr
    x += rausch * rng.standard_normal(n)
    return pol * x


def _hf_effekt(h):
    return 0.003 * (h - 70.0) + 0.00010 * (h - 70.0) ** 2


def _q_kunst(n, beta, rng, sd=0.03, drift=0.0, mrn_gruppe=1, hf_kopplung=0.0, beta_s=0.5):
    """Synthetische Primaer-Paartabelle (alle Spalten, die primaer_kette und die Auswertung erwarten)."""
    i = np.arange(n)
    dk = rng.normal(-0.2, 1.2, n)
    ka = rng.uniform(3.0, 5.8, n)
    hf_a = rng.uniform(60, 100, n)
    hf_b = np.clip(hf_a + hf_kopplung * dk + rng.normal(-1, 7, n), 35, 180)
    ta_a = 0.25 + rng.normal(0, 0.1, n)
    dT = drift + beta * dk + _hf_effekt(hf_b) - _hf_effekt(hf_a) + rng.normal(0, sd, n)
    st_a = 4.0 + rng.normal(0, 0.5, n)
    dS = beta_s * dk + rng.normal(0, 0.3, n)
    ta = i * 100000.0
    Q = pd.DataFrame({'CSN': i + 1, 'MRN': i // mrn_gruppe + 1, 'split': np.array(['train', 'val', 'test'])[i % 3], 'ta': ta, 'tb': ta + 10800.0, 'ra': ta + 4500.0, 'rb': ta + 15300.0,
                      'ka': ka, 'kb': ka + dk, 'dk': dk, 'gap_h': 3.0, 'zensiert': False, 'ohne_ergebnis': False, 'valid_a': 1.0, 'valid_b': 1.0, 'pol_a': 1.0, 'pol_b': 1.0,
                      'pol_unsicher_a': 0.0, 'pol_unsicher_b': 0.0, 'pol_sicher': True, 'pol_gleich': True, 't_amp_a': ta_a, 't_amp_b': ta_a + dT, 't_roh_a': ta_a, 't_roh_b': ta_a + dT,
                      'steil_a': st_a, 'steil_b': st_a + dS, 'dT': dT, 'dT_roh': dT, 'dS': dS, 'c_a': ta + 600.0, 'c_b': ta + 11400.0, 'w0_a': ta + 570.0, 'w1_a': ta + 630.0,
                      'w0_b': ta + 11370.0, 'w1_b': ta + 11430.0, 'kollision': False, 'hf_a': hf_a, 'hf_b': hf_b, 'dhf': hf_b - hf_a, 'beh_kat': 'kein_hinweis',
                      'dna': rng.normal(0, 3, n), 'dhco3': rng.normal(0, 3, n), 'age': 60.0, 'sex': 1})
    return Q


def _d_kunst(n, rng, pk_slope=0.02, wh_sd=0.02, k_hoch=0, hco3_slope=-0.58, qrs_sd=8.0):
    k = rng.uniform(3.0, 6.0, n)
    if k_hoch:
        k[:k_hoch] = rng.uniform(6.5, 8.0, k_hoch)
    hco3 = rng.normal(22, 5, n)
    t = 0.2 + pk_slope * (k - 4.5) + rng.normal(0, 0.1, n)
    return pd.DataFrame({'CSN': np.arange(n) + 1, 'MRN': np.arange(n) + 1, 'split': 'train', 't_order': np.arange(n) * 1000.0, 'k': k, 'cens': False, 'serum': True, 'valid': 1.0,
                         'pol': 1.0, 'pol_unsicher': 0.0, 't_amp': t, 'hf': rng.normal(80, 12, n), 'age': rng.uniform(30, 85, n), 'sex_m': rng.integers(0, 2, n).astype(float),
                         'hco3': hco3, 'qrs_ms': 100 + hco3_slope * (hco3 - 22) + 1.0 * (k - 6) + rng.normal(0, qrs_sd, n), 'c': np.arange(n) * 1000.0 + 600.0,
                         'valid_rep': 1.0, 'pol_rep': 1.0, 'pol_unsicher_rep': 0.0, 't_amp_rep': t + rng.normal(0, wh_sd, n)})


def _cfg_test(**kw):
    c = copy.deepcopy(CFG)
    c['bootstrap_n'] = 200
    c['nk2_n'] = 60
    for k, v in kw.items():
        if isinstance(v, dict):
            c[k].update(v)
        else:
            c[k] = v
    return c


def _wfdb(root, csn, sno, t0_abs, x, units='mV'):
    import wfdb
    d = Path(root) / str(csn)[-3:] / str(csn) / 'II'
    d.mkdir(parents=True, exist_ok=True)
    bt = dt.datetime.fromtimestamp(t0_abs, dt.timezone.utc)
    wfdb.wrsamp(f'{csn}_{sno}', fs=500, units=[units], sig_name=['II'], p_signal=np.asarray(x).reshape(-1, 1), fmt=['16'], write_dir=str(d), base_time=bt.time(), base_date=bt.date())
    return (sno, 500.0, len(x), float(t0_abs), float(t0_abs + len(x) / 500.0))


def _freigabe(td, h=None):
    h = h or {'spezifikation': 'a' * 8, 'skript': 'b', 'abhaengig': {'x.py': 'c'}, 'konfiguration': 'd', 'vertrag': 'v', 'pipeline': PIPELINE}
    lock = Path(td) / 'lock.json'
    lock.write_text(json.dumps({'hashes': h}))
    return pruefe_start(lock, lambda: dict(h)), lock, h


def kunst_datensatz(root, n_visits=40, seed=7, beta=0.02, n_beh=4):
    """Synthetischer MC-MED-Ausschnitt (CSV und WFDB, 500 Hz, mV): K-abhaengige T-Amplitude. Sonderfaelle: Besuch 1 Dialyse, Besuch 5 Kalium in falscher Einheit,
    Besuch 6 ohne Ergebniszeit, Besuche < n_beh Altinsulin, Besuch 0 negative Polaritaet."""
    root = Path(root)
    (root / 'data').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = dt.datetime(2250, 3, 1, 10, 0, 0, tzinfo=dt.timezone.utc).timestamp()
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    vis, lab, orde, num, spl = [], [], [], [], {'train': [], 'val': [], 'test': []}
    for v in range(n_visits):
        csn = 98000000 + v * 37 + 11
        mrn = 99000000 + v * 53 + 5
        t0 = base + v * 90000.0
        o1, o2 = t0 + 600.0, t0 + 600.0 + rng.uniform(2.0, 5.0) * 3600
        k1 = rng.uniform(3.4, 5.6)
        k2 = float(np.clip(k1 + rng.normal(-0.1, 1.1), 2.6, 5.9))
        vis.append((mrn, csn, 60 + v % 20, 'M' if v % 2 else 'F', iso(t0 - 600), iso(o2 + 8 * 3600)))
        for j, (o, k) in enumerate(((o1, k1), (o2, k2))):
            r = o + 4500
            for comp, val, unit in (('POTASSIUM', f'{k:.1f}', 'mEq/L' if (v == 5 and j == 1) else 'mmol/L'), ('SODIUM', f'{rng.normal(139, 3):.0f}', 'mmol/L'),
                                    ('CO2', f'{rng.normal(23, 3):.0f}', 'mmol/L'), ('GLUCOSE', '110', 'mg/dL'), ('CALCIUM', '9.2', 'mg/dL')):
                lab.append((csn, iso(o), '' if (v == 6 and j == 0) else iso(r), comp, val, unit))
        if v < n_beh:
            orde.append((csn, 'Medications', 'INSULIN REGULAR HUMAN 100 UNIT/ML INJ SOLN', iso(o1 + 600), iso(o1 + 3000)))
        if v == 1:
            orde.append((csn, 'Nursing', 'HEMODIALYSIS', iso(o1 + 600), ''))
        orde.append((csn, 'Medications', 'ASPIRIN 81 MG PO TABS', iso(o1 + 100), iso(o1 + 200)))
        for o in (o1, o2):
            for m in range(-1, 8):
                num.append((csn, 'Monitor', 'HR', 70 + rng.normal(0, 1.5), iso(o + 60 * m)))
        spl['test' if v % 10 == 9 else ('val' if v % 5 == 0 else 'train')].append(csn)
        for sno, (o, k) in enumerate(((o1, k1), (o2, k2)), start=1):
            x = kunst_ecg(420.0, hr=70 + rng.normal(0, 2), t_amp=0.20 + beta * (k - 4.5), pol=-1 if v % 6 == 0 else 1, rng=rng)
            _wfdb(root / 'data' / 'waveforms', csn, sno, o - 120, x)
    pd.DataFrame(vis, columns=['MRN', 'CSN', 'Age', 'Gender', 'Roomed_time', 'Departure_time']).to_csv(root / 'data' / 'visits.csv', index=False)
    pd.DataFrame(lab, columns=['CSN', 'Order_time', 'Result_time', 'Component_name', 'Component_value', 'Component_units']).to_csv(root / 'data' / 'labs.csv', index=False)
    pd.DataFrame(orde, columns=['CSN', 'Order_type', 'Procedure_name', 'Order_time', 'First_admin_time']).to_csv(root / 'data' / 'orders.csv', index=False)
    pd.DataFrame(num, columns=['CSN', 'Source', 'Measure', 'Value', 'Time']).to_csv(root / 'data' / 'numerics.csv', index=False)
    for nm, l in spl.items():
        pd.Series(l, dtype='int64').to_csv(root / 'data' / f'split_random_{nm}.csv', index=False, header=False)
    return {'n': n_visits, 'beta': beta, 'test': spl['test']}


# =========================================================================== Selbsttest
def selbsttest():
    import traceback
    ergebnisse = []

    def T(name):
        def deko(fn):
            t_0 = time.time()
            try:
                fn()
                ergebnisse.append((name, True, ''))
            except Exception as e:
                ergebnisse.append((name, False, ''.join(traceback.format_exception_only(type(e), e)).strip() + ' @ ' + str(traceback.extract_tb(e.__traceback__)[-1].lineno)))
            print(('OK    ' if ergebnisse[-1][1] else 'FEHLER'), name, '(%d s)' % (time.time() - t_0), ('' if ergebnisse[-1][1] else ' :: ' + ergebnisse[-1][2]), flush=True)
            return fn
        return deko

    def muss_scheitern(exc, fn, text=''):
        try:
            fn()
        except exc:
            return
        raise AssertionError('kein Abbruch: ' + text)

    @T('Maske nach Zaehlfeldern (auch Gleitkomma), Seeds und Schaetzer unberuehrt')
    def _():
        m = mask_tree({'n': 5, 'paare_x': 7.0, 'seed': 5, 'beta': 3.0, 'personen': [1, 2, 30], 'sub': {'n_x': 9, 'wert': 4}, 'ok': 12, 'null': 0, 'nan': float('nan')})
        assert m == {'n': '<10', 'paare_x': '<10', 'seed': 5, 'beta': 3.0, 'personen': ['<10', '<10', 30], 'sub': {'n_x': '<10', 'wert': 4}, 'ok': 12, 'null': 0, 'nan': None}, m

    @T('Zaehlkette: Komplemente vergroebert, auch Folgeschritte gleicher Zahl')
    def _():
        z = zaehlkette_vergroebern({'a': 100, 'b': 95, 'c': 95, 'd': 50, 'e': 45})
        assert z['a'] == 100 and z['b'] == 90 and z['c'] == 90 and z['d'] == 50 and z['e'] == 40, z
        assert zaehlkette_vergroebern({'a': 100, 'b': 80}) == {'a': 100, 'b': 80}

    @T('Behandlungsklassen inklusive Dialyse und Calcium')
    def _():
        k = behandlung_klasse
        assert k('INSULIN REGULAR HUMAN 100 UNIT/ML INJ SOLN') == 'insulin_regular' and k('INSULIN LISPRO SC') == 'insulin_andere'
        assert k('CALCIUM GLUC IN NACL, ISO-OSM 1 GRAM/50 ML IV SOLN') == 'calcium_iv' and k('CALCIUM CARBONATE 500 MG PO TABS') is None
        assert k('HEMODIALYSIS') == 'dialyse' and k('DIALYZE PATIENT') == 'dialyse' and k('CAPD PERITONEAL DIALYSIS') == 'dialyse'
        assert k('PERIT.DIALYSIS 6-1.5 % DEXTROS CA 2.5 MEQ/L IP SOLN') == 'dialyse' and k('DEXTROSE-SOD CITRATE-CITRIC AC (CRRT - RN TITRATION)') == 'dialyse'
        assert k('DIALYSIS ACCESS') is None and k('CONSULT TO NEPHROLOGY') is None
        assert k('SODIUM BICARBONATE 8.4 % (1 MEQ/ML) IV SYRG') == 'bikarbonat_iv' and k('SODIUM BICARBONATE 650 MG PO TABS') is None
        assert k('POTASSIUM CHLORIDE 20 MEQ PO TBTQ') == 'kalium_gabe' and k('FUROSEMIDE 10 MG/ML INJ SOLN') == 'schleifendiuretikum' and k('ASPIRIN 81 MG') is None
        assert 'calcium_iv' not in CFG['kaliumsenkend'] and 'calcium_iv' in CFG['ecg_wirksam'] and 'calcium_iv' in CFG['dok_klassen']

    @T('Behandlungskategorien: Raender (>=), Calcium, Dialyse, Gabe ohne Zeit, ECG-Zeiten erweitern das Fenster')
    def _():
        S0 = 10000.0
        P = pd.DataFrame({'CSN': [1], 'ta': [S0], 'tb': [20000.0], 'rb': [24500.0], 'c_a': [10600.0], 'c_b': [20600.0]})

        def kat(rows, P_=P):
            trt = pd.DataFrame(rows, columns=['CSN', 'cls', 't_admin', 't_ord'])
            return behandlung_kategorie(P_, trt).iloc[0]

        E = 24500.0
        assert kat([(1, 'insulin_regular', S0 - 7200, np.nan)]).beh_kat == 'dokumentiert'                  # Rand S - 120 min eingeschlossen
        assert kat([(1, 'insulin_regular', S0 - 7201, np.nan)]).beh_kat == 'unklar'                         # davor: frueher, fortwirkend
        assert kat([(1, 'insulin_regular', S0 - 21600, np.nan)]).beh_kat == 'unklar'
        assert kat([(1, 'insulin_regular', S0 - 21601, np.nan)]).beh_kat == 'kein_hinweis'
        assert kat([(1, 'insulin_regular', E, np.nan)]).beh_kat == 'dokumentiert'                           # Rand E eingeschlossen
        assert kat([(1, 'insulin_regular', E + 1, np.nan)]).beh_kat == 'kein_hinweis'
        r = kat([(1, 'calcium_iv', S0 + 100, np.nan)])
        assert r.beh_kat == 'dokumentiert' and r.n_ca == 1 and r.n_kl == 0
        assert kat([(1, 'kalium_gabe', S0 + 5, np.nan)]).beh_kat == 'dokumentiert'
        assert kat([(1, 'insulin_regular', np.nan, S0)]).beh_kat == 'unklar'                                # Auftrag ohne Gabezeit
        assert kat([(1, 'schleifendiuretikum', S0 + 5, np.nan)]).beh_kat == 'unklar'
        assert kat([(1, 'dialyse', np.nan, S0 - 21600)]).beh_kat == 'dokumentiert'                          # Dialyse-Auftrag im Fenster
        assert kat([(1, 'dialyse', np.nan, S0 - 21601)]).beh_kat == 'unklar'                                # Dialyse sonst im Besuch
        assert kat([(2, 'insulin_regular', S0, np.nan)]).beh_kat == 'kein_hinweis'                          # anderer Besuch
        # ECG-Zeit erweitert das Fenster: ohne Ergebniszeit, Gabe nach dem zweiten Auftrag aber vor der zweiten ECG-Mitte
        P2 = pd.DataFrame({'CSN': [1], 'ta': [S0], 'tb': [20000.0], 'rb': [np.nan], 'c_a': [10600.0], 'c_b': [20600.0]})
        assert kat([(1, 'insulin_regular', 20500.0, np.nan)], P2).beh_kat == 'dokumentiert'
        assert kat([(1, 'insulin_regular', 20700.0, np.nan)], P2).beh_kat == 'kein_hinweis'
        # ECG vor dem ersten Auftrag verschiebt den Fensterbeginn
        P3 = pd.DataFrame({'CSN': [1], 'ta': [S0], 'tb': [20000.0], 'rb': [24500.0], 'c_a': [9000.0], 'c_b': [20600.0]})
        assert kat([(1, 'insulin_regular', 9000 - 7200, np.nan)], P3).beh_kat == 'dokumentiert' and kat([(1, 'insulin_regular', 9000 - 7201, np.nan)], P3).beh_kat == 'unklar'
        assert kat([]).beh_kat == 'kein_hinweis'

    @T('Polaritaet: positiv, negativ, zweiphasig unsicher')
    def _():
        x = kunst_ecg(30, rng=np.random.default_rng(1))
        p, u, n = polaritaet(x)
        assert p == 1 and not u and n >= 20, (p, u, n)
        p2, u2, _n = polaritaet(-x)
        assert p2 == -1 and not u2
        t = np.arange(30 * FS) / FS
        z = np.zeros_like(t)
        for r in np.arange(0.5, 29, 0.9):
            z += 0.6 * np.exp(-0.5 * ((t - r) / 0.010) ** 2) - 0.55 * np.exp(-0.5 * ((t - r - 0.030) / 0.010) ** 2)
        assert polaritaet(z)[1], 'zweiphasig muss unsicher sein'
        assert polaritaet(np.zeros(100))[0] is None

    @T('Messkette v3: negative QRS-Richtung gemessen (AW-203 allein lehnt ab); relative und rohe T-Amplitude')
    def _():
        M, V = chain()
        xp = kunst_ecg(80, t_amp=0.30, rng=np.random.default_rng(2))
        r_alt = V.measure_window_v2(-xp, 1000.0, 1008.0, 1068.0, CFG['messkette']['hf_schwelle'])
        assert r_alt['valid'] == 0 and r_alt['reason'] == 'qrs_negativ'
        rp, rn = messe_fenster(xp, 1000.0, 1008.0, 1068.0), messe_fenster(-xp, 1000.0, 1008.0, 1068.0)
        assert rp['valid'] == 1 and rn['valid'] == 1 and rp['pol'] == 1 and rn['pol'] == -1
        assert abs(rp['t_amp'] - rn['t_amp']) < 0.02 and 0.2 < rp['t_amp'] < 0.4 and rp['steil'] > 0
        assert abs(rn['t_roh'] + rn['t_amp']) < 1e-12 and abs(rp['t_roh'] - rp['t_amp']) < 1e-12          # roh = pol * relativ
        lo = messe_fenster(kunst_ecg(80, t_amp=0.15, rng=np.random.default_rng(3), pol=-1), 0.0, 8, 68)
        hi = messe_fenster(kunst_ecg(80, t_amp=0.35, rng=np.random.default_rng(4), pol=-1), 0.0, 8, 68)
        assert hi['t_amp'] - lo['t_amp'] > 0.12

    @T('D3-Grenzen: Obergrenze 3 h, Ergebnis vor Auftrag, ohne Ergebniszeit kein D3 (keine 60-Minuten-Ergaenzung)')
    def _():
        assert d3_grenzen(1000.0, 1000.0 + 4500) == (100.0, 1000.0 + 4500 + 900)
        lo, hi = d3_grenzen(0.0, 20000.0)
        assert lo == -900.0 and hi == 10800.0, (lo, hi)                        # min(20000+900, 0+10800)
        assert d3_grenzen(5000.0, 4400.0) == (4400.0 - 900.0, 5000.0 + 900.0)
        assert d3_grenzen(0.0, None) is None and d3_grenzen(0.0, float('nan')) is None

    @T('Messfenster an die tatsaechlichen Segmentgrenzen verschoben')
    def _():
        assert messfenster_um(10.0, 0.0, 200.0) == (0.0, 60.0) and messfenster_um(195.0, 0.0, 200.0) == (140.0, 200.0)
        assert messfenster_um(100.0, 0.0, 200.0) == (70.0, 130.0) and messfenster_um(30.0, 0.0, 50.0) is None

    @T('Einheitenvertrag: mV Pflicht, uV wird abgelehnt; WFDB-Kopf als UTC; CSV-Zeit mit Z als UTC; Zeitpruefung')
    def _():
        einheit_pruefen(['mV'])
        muss_scheitern(ValueError, lambda: einheit_pruefen(['uV']), 'uV')
        with tempfile.TemporaryDirectory() as td:
            t0 = dt.datetime(2250, 3, 1, 10, 0, 0, tzinfo=dt.timezone.utc).timestamp()
            x = kunst_ecg(20, rng=np.random.default_rng(5))
            seg = _wfdb(td, 98000001, 1, t0, x)
            ns, fs, n, tk, einheit = lies_kopf(Path(td) / '001' / '98000001' / 'II' / '98000001_1.hea')
            assert abs(tk - t0) < 1e-6 and fs == 500.0 and n == len(x) and einheit == 'mV'
            r = lies_roh(98000001, seg, t0, t0 + 20, td)
            assert r is not None and abs(r[0] - t0) < 1e-6
            seg_u = _wfdb(td, 98000002, 1, t0, x, units='uV')
            muss_scheitern(ValueError, lambda: lies_roh(98000002, seg_u, t0, t0 + 20, td), 'uV-Datei')
            import duckdb
            c = duckdb.connect()
            assert c.execute("select epoch_us(try_cast('2250-03-01T10:00:00Z' AS TIMESTAMP))/1e6").fetchone()[0] == t0
            c.execute('CREATE TABLE hrmon(CSN BIGINT, t DOUBLE, hr DOUBLE)')
            c.execute('CREATE TABLE seg(CSN BIGINT, seg INT, t0 DOUBLE)')
            for i in range(25):
                c.execute('INSERT INTO seg VALUES (?,1,?)', [i, 1000.0 + i * 5000])
                c.execute('INSERT INTO hrmon VALUES (?,?,70)', [i, 1040.0 + i * 5000])
            assert abs(zeitvertrag_pruefen(c) - 40.0) < 1e-9
            c.execute('UPDATE hrmon SET t = t + 10800')
            muss_scheitern(ValueError, lambda: zeitvertrag_pruefen(c), '3 h Versatz')
            c.execute('UPDATE hrmon SET t = t - 10800')
            c.execute('DELETE FROM seg WHERE CSN >= 10')
            muss_scheitern(ValueError, lambda: zeitvertrag_pruefen(c), 'zu wenige Besuche')

    @T('Globale Blocksuche ueber Segmente: D1 unabhaengig vom D3-Cache, Wiederholung >= 90 s ueber Segmente, gestoerte Mitte')
    def _():
        with tempfile.TemporaryDirectory() as td:
            csn = 98000010
            t0 = 100000.0
            rng = np.random.default_rng(6)
            x1 = kunst_ecg(300, rng=rng)
            x2 = kunst_ecg(300, rng=rng)
            s1 = _wfdb(td, csn, 1, t0, x1)                # [t0, t0+300]
            s2 = _wfdb(td, csn, 2, t0 + 250, x2)          # ueberlappt: [t0+250, t0+550]
            x1b = x1.copy()
            x1b[130 * FS:200 * FS] = 0.0                  # Stoerung in Segment 1 um t0+165
            s1b = _wfdb(td, csn, 3, t0 + 2000, x1b)       # eigenes Segment: [t0+2000, t0+2300]
            b = naechster_block(csn, [s1, s2], t0 + 150, t0, t0 + 550, wave=td)
            assert b is not None and abs(b[1] - (t0 + 150)) <= 5.0 and b[2][0] == 1, b
            # D1 sucht global: Referenz ausserhalb von Segment 1, naechster saubere Block im Segment 2
            b2 = naechster_block(csn, [s1, s2], t0 + 520, t0 + 520 - 900, t0 + 520 + 900, wave=td)
            assert b2 is not None and b2[2][0] == 2 and abs(b2[1] - (t0 + 520)) <= 30.0, b2
            # Wiederholung: Mitten weniger als 90 s von der ersten Mitte werden uebersprungen, auch aus dem anderen (ueberlappenden) Segment
            c1 = t0 + 275.0
            b3 = naechster_block(csn, [s1, s2], c1, t0, t0 + 550, ausser=[c1], min_abstand=90.0, wave=td)
            assert b3 is not None and abs(b3[1] - c1) >= 90.0, b3
            # gestoerte Mitte wird uebersprungen
            b4 = naechster_block(csn, [s1b], t0 + 2165, t0 + 2000, t0 + 2300, wave=td)
            assert b4 is not None and abs(b4[1] - (t0 + 2165)) >= 30.0, b4
            assert naechster_block(csn, [s1], t0 + 150, t0 + 400, t0 + 500, wave=td) is None

    @T('NK1: tatsaechlich gewaehlter Block geprueft (Abstand 2 bis 6 h, Fenster ausserhalb D3)')
    def _():
        cfg = CFG
        to, lo, hi = 100000.0, 100000.0 - 900, 100000.0 + 5400
        assert shift_gueltig(to + 7200, to + 7170, to + 7230, to, lo, hi, cfg)
        assert not shift_gueltig(to + 7199, to + 7170, to + 7230, to, lo, hi, cfg)
        assert not shift_gueltig(to + 21601, to + 21570, to + 21630, to, lo, hi, cfg)
        assert shift_gueltig(to - 7200, to - 7230, to - 7170, to, lo, hi, cfg)
        lo2, hi2 = to - 900, to + 10800 + 900                  # D3 reicht weit: Fenster ueberlappt D3
        assert not shift_gueltig(to + 7200, to + 7170, to + 7230, to, lo2, hi2, cfg)

    @T('Paarbildung: Abstand 1 bis 12 h, konsekutiv, Flags ohne_ergebnis und zensiert (kein Ergaenzen)')
    def _():
        h = 3600.0
        D = pd.DataFrame({'CSN': [1, 1, 1, 1, 2, 2, 3, 3], 'MRN': [1, 1, 1, 1, 2, 2, 3, 3], 'split': 'train', 't_order': [0, 0.5 * h, 3 * h, 17 * h, 0, 2 * h, 0, 2 * h],
                          't_result': [h, 1.5 * h, 4 * h, 18 * h, h, np.nan, h, 3 * h], 'k': [4.0, 4.5, 5.5, 5.0, 4.0, 3.0, 4.0, 10.0],
                          'cens': [False] * 7 + [True], 'na': 140.0, 'hco3': 22.0, 'glc': 100.0, 'ca': 9.0, 'age': 60.0, 'sex': 1})
        P = paare_bauen(D)
        assert len(P) == 3, P
        p1 = P[P.CSN == 1].iloc[0]
        assert p1.ta == 0.5 * h and p1.tb == 3 * h and not p1.ohne_ergebnis and not p1.zensiert           # (0 -> 0,5 h) < 1 h entfaellt; (3 h -> 17 h) 14 h entfaellt
        assert P[P.CSN == 2].iloc[0].ohne_ergebnis and not P[P.CSN == 2].iloc[0].zensiert
        assert P[P.CSN == 3].iloc[0].zensiert and not P[P.CSN == 3].iloc[0].ohne_ergebnis

    @T('Serum-Zuordnung (kein Verlust bei Doppelwerten), Zensur, Einheiten, H3 nur Serum')
    def _():
        import duckdb
        with tempfile.TemporaryDirectory() as td:
            fr, lock, h = _freigabe(td)
            c = duckdb.connect()
            c.execute('CREATE TABLE meta(freigabe_hash VARCHAR, test_geladen BOOLEAN, vertrag VARCHAR)')
            c.execute('INSERT INTO meta VALUES (?,?,?)', [fr.hash, False, 'v'])
            c.execute('CREATE TABLE visit(CSN BIGINT, MRN BIGINT, split VARCHAR, Age DOUBLE, Gender VARCHAR, r DOUBLE, d DOUBLE)')
            c.execute("INSERT INTO visit VALUES (1, 11, 'train', 60, 'M', 0, 0)")
            c.execute('CREATE TABLE labs(CSN BIGINT, t_order DOUBLE, t_result DOUBLE, cname VARCHAR, cval VARCHAR, unit VARCHAR)')
            rows = [(1, 100.0, 4000.0, 'POTASSIUM', '5.0', 'mmol/L'), (1, 100.0, 4000.0, 'POC:POTASSIUM, ISTAT', '5.2', 'mmol/L'),   # Serum und POC gleichzeitig
                    (1, 200.0, 100.0, 'POC:POTASSIUM, ISTAT', '4.0', 'mmol/L'),                                                      # nur POC
                    (1, 300.0, 4000.0, 'POTASSIUM', '4.0', 'mmol/L'), (1, 300.0, 4000.0, 'POTASSIUM', '4.1', 'mmol/L'),              # widerspruechlich
                    (1, 400.0, 4000.0, 'POTASSIUM', '>10.0', 'mmol/L'), (1, 500.0, 4000.0, 'POTASSIUM', '<2.0', 'mmol/L'),            # zensiert
                    (1, 600.0, 4000.0, 'POTASSIUM', '4.4', 'mEq/L'),                                                                  # falsche Einheit
                    (1, 100.0, 4000.0, 'SODIUM', '140', 'mmol/L'), (1, 100.0, 4000.0, 'CO2', '22', 'mmol/L')]
            for r in rows:
                c.execute('INSERT INTO labs VALUES (?,?,?,?,?,?)', list(r))
            D = entnahmen_tabelle(c, fr)
            Ds, Da = draws_serum(D), draws_alle(D)
            assert sorted(Ds.t_order) == [100.0, 400.0, 500.0] and Ds[Ds.t_order == 100.0].iloc[0].k == 5.0 and Ds.serum.all()
            assert Ds[Ds.t_order == 400.0].iloc[0].cens and Ds[Ds.t_order == 500.0].iloc[0].cens
            a1 = Da[Da.t_order == 100.0].iloc[0]
            assert a1.serum and a1.k == 5.0, 'Serumstatus darf bei gleichzeitigem POC-Wert nicht verloren gehen'
            assert (not Da[Da.t_order == 200.0].iloc[0].serum) and 300.0 not in set(Da.t_order) and 600.0 not in set(Da.t_order)
            assert a1.na == 140.0 and a1.hco3 == 22.0
            # Einheitenzaehlung braucht seg
            c.execute('CREATE TABLE seg(CSN BIGINT, seg INT, fs DOUBLE, n BIGINT, t0 DOUBLE, t1 DOUBLE, einheit VARCHAR)')
            c.execute("INSERT INTO seg VALUES (1,1,500,1000,0,2,'mV'), (1,2,500,1000,0,2,'uV')")
            assert einheiten_zaehlung(c) == {'kalium_zeilen_andere_einheit': 1, 'segmente_andere_signaleinheit': 1}
            # H3 nimmt nur Serum und keine zensierten Werte
            Dm = _d_kunst(400, np.random.default_rng(7), k_hoch=200)
            Dm.loc[:99, 'serum'] = False
            Dm.loc[100:149, 'cens'] = True
            R = h3_auswertung(Dm, _cfg_test())
            assert R['n'] == 50, R['n']
            muss_scheitern(PermissionError, lambda: entnahmen_tabelle(c, 'ungueltig'), 'ungueltige Freigabe')
            fr2, _l, _h = _freigabe(td, {**h, 'skript': 'anders'})
            muss_scheitern(PermissionError, lambda: entnahmen_tabelle(c, fr2), 'Bestand unter anderer Freigabe geladen')

    @T('Primaerkette: Ausschluesse einzeln gezaehlt, Kollision, Polaritaetsgruppen')
    def _():
        Q = _q_kunst(40, 0.02, np.random.default_rng(13))
        Q.loc[0, 'ohne_ergebnis'] = True
        Q.loc[1, 'zensiert'] = True
        Q.loc[2, 'c_a'] = np.nan
        Q.loc[3, 'valid_b'] = 0.0
        Q.loc[4, ['pol_b', 'pol_gleich']] = [-1.0, False]
        Q.loc[5, ['pol_unsicher_a', 'pol_sicher', 'pol_gleich']] = [1.0, False, False]
        Q.loc[6, 'kollision'] = True
        Q.loc[7, 'hf_b'] = np.nan
        Q.loc[8, 'beh_kat'] = 'dokumentiert'
        Q.loc[9, 'beh_kat'] = 'unklar'
        Q.loc[10, 'ka'] = 6.4
        prim, z, g = primaer_kette(Q)
        erw = {'paare_konsekutiv': 40, 'nach_ausschluss_ohne_ergebniszeit': 39, 'nach_ausschluss_zensiert': 38, 'ecg_zeitlich_beide_enden': 37, 'messung_gueltig_beide_enden': 36,
               'polaritaet_sicher_und_gleich': 34, 'ohne_fensterueberlappung': 33, 'hf_beide_enden': 32, 'davon_behandlung_dokumentiert': 1, 'davon_behandlung_unklar': 1,
               'behandlung_zugelassen': 30, 'davon_erstwert_ab_6': 1, 'primaerpaare': 29}
        for k, v in erw.items():
            assert z[k] == v, (k, z[k], v)
        assert g['pol_mm'] == 0 and g['pol_gemischt'] == 1 and g['pol_unsicher'] == 1 and g['pol_pp'] == 34
        p2, z2, _g = primaer_kette(Q, beh_ok=('kein_hinweis', 'unklar'))
        assert z2['primaerpaare'] == 30

    @T('HF-Spline: Differenz f(HF2)-f(HF1), nichtlinearer HF-Effekt, unadjustiert und adjustiert auf identischen Faellen')
    def _():
        B = spline_differenz(np.array([70.0, 70.0]), np.array([70.0, 90.0]))
        assert B.shape == (2, 3) and np.allclose(B[0], 0) and not np.allclose(B[1], 0)
        r = np.random.default_rng(14)
        Q = _q_kunst(500, 0.02, r, hf_kopplung=-4.0)               # HF steigt, wenn Kalium faellt: Verzerrung ohne Adjustierung
        q, adj, unadj = h1_modelle(Q)
        assert adj['n'] == unadj['n'] == len(q) == 500, 'identische Faelle'
        assert abs(adj['beta'] - 0.02) < 0.004, adj
        assert abs(unadj['beta'] - 0.02) > 0.004, 'Verzerrung durch HF-Kopplung muss ohne Adjustierung sichtbar sein'

    @T('H1: freier Achsenabschnitt (Drift), Sensitivitaeten, Ausgangsbereich, Anstieg/Abfall, rohe Amplitude, Huber, Einfluss')
    def _():
        cfg = _cfg_test()
        Q = _q_kunst(420, 0.02, np.random.default_rng(15), drift=0.03)
        Q.loc[:9, 'dT'] += 1.5                                         # Ausreisser
        Q['t_amp_b'] = Q.t_amp_a + Q.dT
        Q['t_roh_b'], Q['dT_roh'] = Q.t_amp_b, Q.dT
        R = h1_auswertung(Q, cfg)
        e = R['primaer']
        assert e['beurteilbar'] and e['n'] == 420, e
        S = R['sensitivitaeten']
        assert abs(S['huber']['beta'] - 0.02) < 0.006 and abs(S['huber']['beta'] - 0.02) < abs(e['beta'] - 0.02), (S['huber'], e)       # robust gegen Ausreisser
        assert S['einfluss']['n_einflussreich'] >= 5 and S['einfluss']['ohne_einflussreiche']['beurteilbar']
        for k in ('k1_bis_3.5', 'k1_3.5_bis_5.0', 'k1_ueber_5.0'):
            assert k in S['ausgangsbereich']
        assert S['nur_anstieg']['n'] + S['nur_abfall']['n'] <= 420 and S['nur_abfall']['n'] > S['nur_anstieg']['n']
        for k in ('mit_k1', 'mit_bikarbonat', 'nur_abs_dhf_unter_10', 'eine_beobachtung_je_person', 'gemischtes_modell', 'ohne_dokumentierte_behandlung', 'alle_erstwerte',
                  'roh_t_amplitude_primaerpaare', 'roh_t_amplitude_alle_gueltigen_paare_inkl_polaritaetswechsel', 'messfehler_kalium'):
            assert k in S, k
        assert R['unadjustiert_gleiche_faelle']['n'] == e['n']
        assert R['kategorie'] in ('positiv_ueber_grenze', 'positiv_grenze_offen', 'positiv_unter_grenze', 'unpraezise', 'grenze_ausgeschlossen'), R['kategorie']
        assert 'zaehlkette_roh' in R and R['polaritaetsgruppen']['pol_pp'] == 420 and 'timing' in R and 'praezision' in R
        # ohne Achsenabschnitt waere die Steigung bei Mittelwert dk != 0 verzerrt; mit freiem Achsenabschnitt nicht
        Q2 = _q_kunst(420, 0.02, np.random.default_rng(16), drift=0.05)
        assert abs(h1_modelle(Q2)[1]['beta'] - 0.02) < 0.004

    @T('Kategorien: ueberschneidungsfrei und vollstaendig; Nullnaehe nur ueber Aequivalenzgrenzen')
    def _():
        zul = {'negativ', 'positiv_ueber_grenze', 'positiv_unter_grenze', 'positiv_grenze_offen', 'grenze_ausgeschlossen', 'unpraezise'}
        for lo in np.arange(-0.03, 0.04, 0.0025):
            for hi in np.arange(lo, 0.05, 0.0025):
                k = kategorie_intervall({'beurteilbar': True, 'ci_lo': lo, 'ci_hi': hi}, 0.010)
                assert k in zul, (lo, hi, k)
        kc = lambda lo, hi, g=0.010: kategorie_intervall({'beurteilbar': True, 'ci_lo': lo, 'ci_hi': hi}, g)
        assert kc(0.012, 0.03) == 'positiv_ueber_grenze' and kc(0.002, 0.008) == 'positiv_unter_grenze' and kc(0.002, 0.02) == 'positiv_grenze_offen'
        assert kc(-0.002, 0.008) == 'grenze_ausgeschlossen' and kc(-0.002, 0.02) == 'unpraezise' and kc(-0.03, -0.01) == 'negativ'
        assert kategorie_intervall({'beurteilbar': True, 'ci_lo': -0.9, 'ci_hi': -0.4}, 0.29, -1) == 'positiv_ueber_grenze'
        assert kategorie_intervall({'beurteilbar': False}, 0.01) == 'nicht_beurteilbar'
        kn = lambda lo, hi: kategorie_nullnaehe({'beurteilbar': True, 'ci_lo': lo, 'ci_hi': hi}, 0.005)
        assert kn(-0.003, 0.004) == 'nahe_null' and kn(-0.02, 0.02) == 'unpraezise' and kn(0.006, 0.02) == 'von_null_verschieden' and kn(0.001, 0.004) == 'nahe_null'

    @T('Gemischtes Modell: Konvergenz, Vorzeichen, Warnungen protokolliert')
    def _():
        Q = _q_kunst(300, 0.02, np.random.default_rng(17), mrn_gruppe=2)
        q, adj, _u = h1_modelle(Q, _cfg_test())
        m = gemischtes_modell(q, 'dT', 'dk', hf_spalten(), adj['beta'])
        assert m['beurteilbar'] and isinstance(m['warnungen'], list) and isinstance(m['konvergiert'], bool) and m['vorzeichen_gleich'] is True
        m2 = gemischtes_modell(q, 'dT', 'dk', hf_spalten(), -adj['beta'])
        assert m2['vorzeichen_gleich'] is False and m2['vertrauenswuerdig'] is False

    @T('Messfehler-Szenarien (Verduennung) und Praezisionsformel')
    def _():
        est = {'beurteilbar': True, 'beta': 0.02, 'ci_lo': 0.01, 'ci_hi': 0.03}
        s = messfehler_szenarien(est, 1.8)
        lam = 1 - 2 * 0.01 / 1.8
        assert abs(s['szenarien'][0]['lambda'] - lam) < 1e-12 and abs(s['szenarien'][0]['beta'] - 0.02 / lam) < 1e-12
        assert messfehler_szenarien(est, 0.02)['szenarien'][0]['beta'] is None
        p = praezisionsplanung(1.3, 0.1, 300, 1.0)
        assert abs(p['se'] - 0.1 / (1.3 * math.sqrt(300))) < 1e-12 and abs(p['mdb80'] - 2.8 * p['se']) < 1e-12
        assert praezisionsplanung(1.3, 0.1, 300, 2.0)['se'] > p['se']

    @T('Klinische Zielgroesse: mm, Wiederholungsgrenze, individuelle Aenderung je Stufe')
    def _():
        assert mm(0.02) == 0.2 and mm(None) is None
        r = np.random.default_rng(18)
        Q = _q_kunst(500, 0.10, r, sd=0.02)
        wh = icc_zweifenster(0.2 + r.normal(0, .1, 400) + r.normal(0, .03, 400), 0.2 + r.normal(0, .1, 400))
        prim, _z, _g = primaer_kette(Q)
        q, adj, _u = h1_modelle(prim)
        k = klinisch_h1(adj, {'beurteilbar': True, 'rc': 0.08}, prim)
        assert abs(k['mm_je_mmol'] - 10 * adj['beta']) < 1e-12 and k['wiederholungsgrenze_mm'] == 0.8 and abs(k['erwartete_aenderung_je_mmol_im_verhaeltnis_zur_wiederholungsgrenze'] - adj['beta'] / 0.08) < 1e-9
        t = k['individuell_je_kaliumstufe']
        assert sum(v['n'] for v in t.values()) == len(prim)
        assert t['dk_ab_1_aufwaerts']['anteil_gleichgerichtet_ueber_wiederholungsgrenze'] > t['dk_betrag_unter_0.5']['anteil_betrag_ueber_wiederholungsgrenze']
        assert klinisch_h1(adj, {'beurteilbar': False}, prim)['wiederholungsgrenze_mv'] is None
        assert klinisch_h1({'beurteilbar': False}, wh, prim) == {'beurteilbar': False}
        ic = icc_zweifenster(np.array([0.1] * 20) + r.normal(0, .02, 20), np.array([0.1] * 20) + r.normal(0, .02, 20))
        assert abs(ic['rc'] - 1.96 * math.sqrt(2) * ic['sw']) < 1e-12

    @T('Keine Tore: PK und Wiederholbarkeit nur Beschreibung; schlechte Wiederholbarkeit sperrt kein Urteil; Untergrenze der Schaetzbarkeit')
    def _():
        cfg = _cfg_test()
        D = _d_kunst(600, np.random.default_rng(19), wh_sd=0.3)
        T_ = {'Q': _q_kunst(300, 0.02, np.random.default_rng(20)), 'Dm': D, 'H4': pd.DataFrame(), 'Qd1': None, 'Qa': None, 'Qsh': None}
        R = gesamt(T_, cfg)
        assert R['h1']['kategorie'] in ('positiv_ueber_grenze', 'positiv_grenze_offen', 'positiv_unter_grenze'), R['h1']['primaer']
        assert 'erfuellt' not in json.dumps(R, default=str) and R['beschreibungen']['wiederholbarkeit']['sw'] > 0.1
        T2 = dict(T_)
        T2['Q'] = _q_kunst(20, 0.02, np.random.default_rng(21))
        R2 = gesamt(T2, cfg)
        assert R2['h1']['kategorie'] == 'nicht_beurteilbar' and not R2['h1']['primaer']['beurteilbar']

    @T('H3 (N3-M): negative Richtung, Terzile, Mindestzahl, Bikarbonat nur als Surrogat')
    def _():
        cfg = _cfg_test()
        R = h3_auswertung(_d_kunst(2000, np.random.default_rng(23), k_hoch=400), cfg)
        assert R['kategorie'] in ('positiv_ueber_grenze', 'positiv_grenze_offen') and abs(R['modell']['beta'] + 0.58) < 0.15, R
        tz = R['qrs_mittel_nach_bikarbonat_terzil']
        assert tz['unteres']['mittel'] > tz['oberes']['mittel'] and 'kein Beweis' not in R['hinweis'] and 'Azidose nicht belegt' in R['hinweis']
        assert h3_auswertung(_d_kunst(2000, np.random.default_rng(24), k_hoch=20), cfg)['kategorie'] == 'nicht_beurteilbar'
        R0 = h3_auswertung(_d_kunst(2000, np.random.default_rng(25), k_hoch=400, hco3_slope=0.0), cfg)
        assert R0['kategorie'] in ('grenze_ausgeschlossen', 'unpraezise', 'negativ', 'positiv_unter_grenze'), R0['kategorie']

    @T('H4: erste Wiederholung im festen Fenster, Gruppen PE/RN/unklar/dokumentiert, Erstkalium und Abstand, Marke ohne Entscheidungsanspruch')
    def _():
        h = 3600.0
        Ds = pd.DataFrame({'CSN': [1, 1, 1, 1, 2, 2, 3, 3], 'MRN': [1, 1, 1, 1, 2, 2, 3, 3], 'split': 'train', 't_order': [0, 0.1 * h, 0.5 * h, 3 * h, 0, 2 * h, 0, 2 * h],
                           't_result': 1.0, 'k': [6.5, 5.9, 6.2, 4.0, 5.9, 5.0, 6.8, 5.0], 'cens': [False] * 6 + [True, False]})
        Hp = h4_paare(Ds)
        assert len(Hp) == 2 and Hp.iloc[0].tb == 0.5 * h and Hp.iloc[0].kb == 6.2 and Hp.iloc[1].ta == 0.5 * h and Hp.iloc[1].tb == 3 * h, Hp   # <15 min uebersprungen, erste Wiederholung; K1 < 6 und zensiert ausgeschlossen
        cfg = _cfg_test()
        r = np.random.default_rng(26)
        n = 240
        H = pd.DataFrame({'CSN': np.arange(n) + 1, 'MRN': np.arange(n) + 1, 'split': 'train', 'ta': 0.0, 'tb': 10800.0, 'ra': 4500.0, 'rb': 15300.0, 'ka': 6.8, 'kb': 6.4, 'zensiert': False,
                          'gap_h': 3.0, 'valid_a': 1.0, 'pol_unsicher_a': 0.0, 't_amp_a': 0.2, 'c_a': 600.0, 'c_b': np.nan, 'ohne_ergebnis': False, 'beh_kat': 'kein_hinweis'})
        H.loc[80:159, 'kb'] = 4.2                                   # RN
        H.loc[160:179, 'kb'] = 5.2                                  # unklar (Folgewert)
        H.loc[180:199, 'beh_kat'] = 'dokumentiert'
        H.loc[200:219, 'beh_kat'] = 'unklar'                        # unklar (Behandlung)
        H.loc[220:239, 'valid_a'] = 0.0                             # ohne gueltiges ECG
        H['t_amp_a'] = np.where(H.kb >= 5.5, 0.38, 0.20) + r.normal(0, 0.05, n)
        R = h4_auswertung(H, cfg)
        assert R['pe'] == 80 and R['rn'] == 80 and R['unklar'] == 40 and R['dokumentiert'] == 20 and R['n_erstwerte_mit_ecg'] == 220, R
        assert R['auc'] > 0.95 and R['auc_intervall_ueber_0.5'] and R['auc_marke_erreicht'] and 'auc_adjustiert_erstkalium_abstand' in R
        assert R['marke_t_mm'] == 3.0 and R['anteil_pe_ueber_marke']['wert'] > 0.9 and R['anteil_rn_unter_marke']['wert'] > 0.9
        assert R['kategorie'] == 'beschrieben' and 'kein Referenzstandard' in R['hinweis'] and 'Pseudohyperkaliaemie-Aussage' in R['hinweis'] and 'Entscheidungsanspruch' in R['hinweis']
        assert 'sensitivitaet' not in json.dumps(R).lower()
        assert h4_auswertung(H.iloc[:100], cfg)['kategorie'] == 'nicht_beurteilbar'
        assert h4_auswertung(pd.DataFrame(), cfg)['kategorie'] == 'nicht_beurteilbar'

    @T('NK2: feste Vollstichprobe, nie die eigene Person, Bijektion, Nullverteilung zentriert')
    def _():
        r = np.random.default_rng(27)
        for gs in (1, 2, 5):
            mrn = np.repeat(np.arange(100 // gs + 1), gs)[:100]
            ps = nk2_permutationen(100, mrn, 50, 1)
            assert ps is not None and len(ps) == 50
            for p in ps:
                assert sorted(p) == list(range(100)) and (mrn[p] != mrn).all(), gs
        assert nk2_permutationen(10, np.array([1] * 9 + [2]), 5, 1) is None and nk2_permutationen(3, np.array([1, 1, 1]), 5, 1) is None
        cfg = _cfg_test()
        prim, _z, _g = primaer_kette(_q_kunst(400, 0.03, r, mrn_gruppe=4))
        R = nk2_auswertung(prim, cfg)
        assert R['n'] == 400 and abs(R['null_mittel']) < 0.005 and R['p_perm'] < 0.05 and R['beta_beobachtet'] > 0.02 and R['kategorie'] in ('nahe_null', 'unpraezise'), R
        assert 'Rechenpipeline' in R['hinweis']
        p0, _z, _g = primaer_kette(_q_kunst(400, 0.0, np.random.default_rng(28), mrn_gruppe=4))
        assert nk2_auswertung(p0, cfg)['p_perm'] > 0.01
        assert nk2_auswertung(prim.assign(MRN=1), cfg)['kategorie'] == 'nicht_beurteilbar'

    @T('NK1: ohne Mindestzahl 100, mit Praezisionsangabe; Leck wird erkannt; Polaritaet/Kollision der versetzten Fenster')
    def _():
        cfg = _cfg_test()
        r = np.random.default_rng(29)
        Q = _q_kunst(400, 0.02, r)
        prim, _z, _g = primaer_kette(Q)

        def sh(P, leck=0.0, n=None):
            S = P.copy()
            S['t_amp_a'] = 0.25 + r.normal(0, 0.1, len(S))
            S['t_amp_b'] = S.t_amp_a + leck * S.dk + r.normal(0, 0.03, len(S))
            S['dT'] = S.t_amp_b - S.t_amp_a
            return S if n is None else S.iloc[:n]

        e = nk1_auswertung(prim, sh(Q), 0.02, cfg)
        assert e['kategorie'] in ('nahe_null', 'unpraezise') and e['halbbreite'] > 0 and e['n'] == 400, e
        assert nk1_auswertung(prim, sh(Q, leck=0.02), 0.02, cfg)['kategorie'] == 'von_null_verschieden'
        e2 = nk1_auswertung(prim, sh(Q, n=60), 0.02, cfg)
        assert e2['n'] == 60 and e2['beurteilbar'] and 'halbbreite' in e2                      # kleine Zahl wird geschaetzt, nicht verworfen
        S = sh(Q)
        S.loc[:99, 'pol_gleich'] = False
        S.loc[100:149, 'kollision'] = True
        S.loc[150:199, 'valid_b'] = 0.0
        assert nk1_auswertung(prim, S, 0.02, cfg)['n'] == 200
        assert nk1_auswertung(prim, None, 0.02, cfg)['kategorie'] == 'nicht_beurteilbar' and nk1_auswertung(prim, sh(Q, n=5), 0.02, cfg)['kategorie'] == 'nicht_beurteilbar'

    @T('Pipeline-Kontrolle Natrium: unabhaengig, gekoppelt (partiell), Artefakt')
    def _():
        cfg = _cfg_test()
        r = np.random.default_rng(30)
        prim, _z, _g = primaer_kette(_q_kunst(500, 0.02, r))
        e = na_kontrolle(prim, cfg)
        assert e['kontrolle_rein'] and e['kategorie'] in ('nahe_null', 'unpraezise') and 'Pipeline-Kontrolle' in e['hinweis']
        Qg = _q_kunst(500, 0.02, r)
        Qg['dna'] = 3.0 * Qg.dk + r.normal(0, 1.0, 500)
        eg = na_kontrolle(primaer_kette(Qg)[0], cfg)
        assert not eg['kontrolle_rein'] and eg['massgebend'] == 'partiell_dk' and eg['kategorie'] in ('nahe_null', 'unpraezise')
        Qa = _q_kunst(500, 0.0, r)
        Qa['dT'] = Qa.dT + 0.01 * Qa.dna
        assert na_kontrolle(primaer_kette(Qa)[0], cfg)['kategorie'] == 'von_null_verschieden'

    @T('Stabilitaet je Split, D1 gegen D3 auf denselben Paaren, alle Quellen')
    def _():
        cfg = _cfg_test()
        Q = _q_kunst(450, 0.02, np.random.default_rng(31))
        prim, _z, _g = primaer_kette(Q)
        st = stabilitaet_je_split(prim, cfg)
        assert set(st) == {'train', 'val', 'test'} and all(v['beurteilbar'] for v in st.values())
        Qd1 = Q.iloc[:300].copy()
        Qd1.loc[:49, 'valid_a'] = 0.0
        d = d1_vergleich(prim, Qd1, cfg)
        assert d['gemeinsame_paare'] == 250 and d['d3']['n'] == 250 and d['d1']['n'] == 250 and d['primaerpaare_d1_gesamt'] == 250
        assert d1_vergleich(prim, None, cfg) == {'beurteilbar': False}
        assert alle_quellen(Q, cfg)['primaerpaare'] == 450

    @T('Deutung: auffaellige Kontrolle macht H1 und H2 nicht deutbar; Kategorien, Zeitsatz, H2 in der Tabelle')
    def _():
        cfg = _cfg_test()
        T_ = {'Q': _q_kunst(420, 0.02, np.random.default_rng(32)), 'Dm': _d_kunst(2000, np.random.default_rng(33), k_hoch=300), 'H4': pd.DataFrame(), 'Qd1': None, 'Qa': None, 'Qsh': None}
        R = gesamt(T_, cfg)
        d = {x[0]: x for x in R['deutung']}
        assert 'H1' in d and 'H2' in d and 'H3' in d and 'H4' in d and 'Abnahmezeit' in d['H1'][2] and d['H1'][1] == R['h1']['kategorie']
        assert 'H1 klinisch' in d and 'keine individuelle Nachweisgrenze' in d['H1 klinisch'][2]
        R['kontrollen']['na_pipeline']['kategorie'] = 'von_null_verschieden'
        d2 = {x[0]: x for x in deutung(R, cfg)}
        assert d2['H1'][1] == 'nicht_deutbar' and d2['H2'][1] == 'nicht_deutbar' and 'na_pipeline' in d2['H1'][2]
        R['kontrollen']['na_pipeline']['kategorie'] = 'nahe_null'
        R['h1']['kategorie'] = 'negativ'
        assert 'technischer Pruefung' in {x[0]: x for x in deutung(R, cfg)}['H1'][2]

    @T('Freigabe und Bindung: ohne Freigabe kein Datenzugriff (auch Train/Val), Test nur ueber pruefe_start, jeder Hash zaehlt')
    def _():
        muss_scheitern(PermissionError, lambda: Freigabe(None, 'x'), 'Freigabe ohne Token')
        muss_scheitern(PermissionError, lambda: Freigabe(object(), 'x'), 'Freigabe mit falschem Token')
        with tempfile.TemporaryDirectory() as td:
            kunst = Path(td) / 'daten'
            kunst_datensatz(kunst, n_visits=12)
            data = kunst / 'data'
            muss_scheitern(PermissionError, lambda: split_csns(data, 'train', None), 'Train ohne Freigabe')
            muss_scheitern(PermissionError, lambda: split_csns(data, 'test', 'echt'), 'Test ohne Freigabe')
            muss_scheitern(PermissionError, lambda: alle_csns(data, None), 'alle ohne Freigabe')
            fr, lock, h = _freigabe(td)
            assert len(split_csns(data, 'test', fr)) == 1 and split_csns(data, 'alle', fr).enthaelt_test and not split_csns(data, 'train', fr).enthaelt_test
            c = verbinden(Path(td) / 'x.duckdb')
            muss_scheitern(PermissionError, lambda: laden(c, data, data / 'waveforms', [98000011], fr), 'laden mit einfacher Liste')
            muss_scheitern(PermissionError, lambda: laden(c, data, data / 'waveforms', split_csns(data, 'alle', fr), None), 'laden ohne Freigabe')
            laden(c, data, data / 'waveforms', split_csns(data, 'alle', fr), fr)
            assert c.execute("select count(*) from visit where split='test'").fetchone()[0] == 1
            c.close()
            for k, v in (('spezifikation', 'z'), ('skript', 'z'), ('konfiguration', 'z'), ('vertrag', 'z'), ('pipeline', 'z'), ('abhaengig', {'x.py': 'anders'})):
                bad = {**h, k: v}
                muss_scheitern(SystemExit, lambda bad=bad: pruefe_start(lock, lambda: dict(bad)), k)
            muss_scheitern(SystemExit, lambda: pruefe_start(Path(td) / 'fehlt.json', lambda: dict(h)), 'ohne Bindung')
        h0 = hashes()
        alt = CFG['h1']['grenze']
        CFG['h1']['grenze'] = alt + 0.001
        try:
            assert hashes()['konfiguration'] != h0['konfiguration']
        finally:
            CFG['h1']['grenze'] = alt
        assert hashes()['konfiguration'] == h0['konfiguration'] and 'vertrag' in h0
        muss_scheitern(SystemExit, lambda: main(['x', 'gesamtlauf']), 'gesamtlauf ohne Bindung')

    @T('Block-Kollision und Timing: gleiche oder ueberlappende Fenster an beiden Paarenden werden erkannt, Abstaende berichtet, ohne Ergebniszeit keine Messung')
    def _():
        P = pd.DataFrame({'CSN': [1, 2, 3], 'MRN': [1, 2, 3], 'split': 'train', 'ta': [0.0, 100000.0, 200000.0], 'tb': [7200.0, 107200.0, 207200.0], 'ra': [4500.0, 104500.0, 204500.0],
                          'rb': [11700.0, 111700.0, 211700.0], 'ka': 4.0, 'kb': 4.5, 'zensiert': False, 'age': 60.0, 'sex': 1, 'na_a': 140.0, 'na_b': 140.0, 'hco3_a': 22.0, 'hco3_b': 22.0,
                          'glc_a': 100.0, 'glc_b': 100.0, 'gap_h': 2.0, 'ohne_ergebnis': False, 'dk': 0.5, 'dna': 0.0, 'dhco3': 0.0, 'dglc': 0.0})
        M = pd.DataFrame({'CSN': [1, 1, 2, 2, 3, 3], 't_order': [0.0, 7200.0, 100000.0, 107200.0, 200000.0, 207200.0],
                          'valid_d3': 1.0, 'pol_d3': 1.0, 'pol_unsicher_d3': 0.0, 't_amp_d3': 0.2, 't_roh_d3': 0.2, 'steil_d3': 4.0, 'qrs_ms_d3': 90.0,
                          'c_d3': [3000.0, 3030.0, 101000.0, 107800.0, 201000.0, 207600.0], 'w0_d3': [2970.0, 3000.0, 100970.0, 107770.0, 200970.0, 207570.0],
                          'w1_d3': [3030.0, 3060.0, 101030.0, 107830.0, 201030.0, 207630.0], 'reason_d3': ''})
        hf = M[['CSN', 't_order']].copy()
        hf['hf'] = 70.0
        Q = messwerte_anfuegen(P, M, 'd3', hf, pd.DataFrame(columns=['CSN', 'cls', 't_admin', 't_ord']))
        assert list(Q.kollision) == [True, False, False], list(Q.kollision)               # Paar 1: ueberlappende Fenster [2970,3030] und [3000,3060]
        prim, z, g = primaer_kette(Q)
        assert z['ohne_fensterueberlappung'] == z['polaritaet_sicher_und_gleich'] - 1 and z['primaerpaare'] == 2
        tb = timing_beschreibung(Q, CFG)
        assert tb['ende_a']['n'] == 3 and tb['ende_a']['abstand_auftrag_min_q05_25_50_75_95'] is None            # unter 10 Paaren keine Quantile
        tq = timing_beschreibung(_q_kunst(50, 0.02, np.random.default_rng(3)), CFG)
        assert abs(tq['ende_a']['abstand_auftrag_min_q05_25_50_75_95'][2] - 10.0) < 1e-9 and abs(tq['ende_a']['abstand_ergebnis_min_q05_25_50_75_95'][2] + 65.0) < 1e-9
        assert tq['ende_a']['anteil_vor_auftrag'] == 0.0 and tq['ende_a']['anteil_nach_ergebnis'] == 0.0
        # ohne Ergebniszeit keine Messung (keine 60-Minuten-Ergaenzung)
        r = messe_entnahme((1, 1000.0, float('nan'), [], {'cfg': CFG}))
        assert r['ohne_ergebnis'] == 1 and r['blk'] == 0 and 'c_d3' not in r
        r2 = messe_entnahme((1, 1000.0, 5500.0, [], {'cfg': CFG}))
        assert r2['ohne_ergebnis'] == 0 and r2['blk'] == 0

    @T('Eingabepruefung gegen SHA256SUMS: Treffer, Abweichung, fehlende Datei')
    def _():
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'ds' / 'data'
            d.mkdir(parents=True)
            for f in EINGABEN:
                (d / f).write_text('inhalt ' + f)
            assert eingaben_pruefen(d) == {'geprueft': False, 'grund': 'SHA256SUMS.txt fehlt'}
            (d.parent / 'SHA256SUMS.txt').write_text(''.join(f'{sha(d / f)}  data/{f}\n' for f in EINGABEN))
            assert eingaben_pruefen(d) == {'geprueft': True, 'dateien': len(EINGABEN)}
            (d / 'labs.csv').write_text('veraendert')
            muss_scheitern(ValueError, lambda: eingaben_pruefen(d), 'Abweichung')
            (d.parent / 'SHA256SUMS.txt').write_text('')
            muss_scheitern(ValueError, lambda: eingaben_pruefen(d), 'fehlender Eintrag')

    @T('Annotation: kaliumblinde Stichprobe (eine je Person, ohne Kalium), Vergleich')
    def _():
        D = _d_kunst(600, np.random.default_rng(34))
        a, schl = annotation_stichprobe(D, 100, 5)
        assert list(a.columns) == ['annot_id'] and len(a) == 100 and a.annot_id.is_unique and set(schl.columns) == {'annot_id', 'CSN', 'c'}
        a2, _s = annotation_stichprobe(D, 100, 5)
        assert (a.annot_id == a2.annot_id).all() and schl.CSN.is_unique
        r = np.random.default_rng(35)
        pipe = pd.DataFrame({'annot_id': a.annot_id, 't_pipeline': 0.2 + r.normal(0, 0.1, 100)})
        user = pd.DataFrame({'annot_id': a.annot_id, 't_anwender': pipe.t_pipeline + r.normal(0, 0.02, 100)})
        e = annotation_auswertung(user, pipe)
        assert e['icc'] > 0.9 and e['anteil_innerhalb_0.05_mv'] > 0.9 and abs(e['bias']) < 0.02

    @T('Annotation vorbereiten: nur Train und Val, Bilder ohne Kalium, Schluessel privat')
    def _():
        global WAVE
        alt_wave = WAVE
        cfg = _cfg_test()
        try:
            with tempfile.TemporaryDirectory() as td:
                info = kunst_datensatz(td, n_visits=14, beta=0.02, n_beh=0)
                data = Path(td) / 'data'
                fr, lock, h = _freigabe(td)
                liste = annotation_vorbereiten(data, data / 'waveforms', Path(td) / 'priv', lock, lambda: dict(h), cfg, 6, Path(td) / 'ann')
                assert len(liste) == 6 and list(liste.columns) == ['annot_id', 'bild'] and all((Path(td) / 'ann' / b).stat().st_size > 1000 for b in liste.bild if b)
                sch = pd.read_csv(Path(td) / 'priv' / 'annotation_schluessel_privat.csv')
                assert not (set(sch.CSN) & set(info['test'])), 'Testbesuch in der Annotation'
                assert open(Path(td) / 'ann' / 'annotation_liste.csv').readline().strip() == 'annot_id,bild'
        finally:
            WAVE = alt_wave

    @T('Ende-zu-Ende gesamtlauf: alle Splits nur nach Bindung, Sonderfaelle, Ausgabe maskiert')
    def _():
        global WAVE
        alt_wave = WAVE
        cfg = _cfg_test(min_schaetzung={'paare': 15, 'personen': 5}, nk={'grenze': 0.005, 'n_min': 5})
        try:
            with tempfile.TemporaryDirectory() as td:
                info = kunst_datensatz(td, n_visits=40, beta=0.02, n_beh=4)
                data = Path(td) / 'data'
                fr, lock, h = _freigabe(td)
                out = gesamtlauf(data, data / 'waveforms', Path(td) / 'priv', lock, lambda: dict(h), cfg, 1, Path(td) / 'aus')
                js = json.dumps(out, default=str)
                assert not re.search(r'(?<![\d.])\d{8,}(?![\d.])', js), 'Kennung in der Ausgabe'
                assert '"CSN"' not in js and '"MRN"' not in js
                assert (Path(td) / 'aus' / 'gesamt_maskiert.json').exists()
                z = out['h1']['zaehlkette']
                assert z['paare_konsekutiv'] >= 30 and z['nach_ausschluss_ohne_ergebniszeit'] < z['paare_konsekutiv'], z
                e = out['h1']['primaer']
                assert e['beurteilbar'] and abs(e['beta'] - 0.02) < 0.012 and e['beta'] > 0, e
                assert out['h1']['kategorie'] in ('positiv_ueber_grenze', 'positiv_grenze_offen', 'positiv_unter_grenze')
                assert set(out['h1']['stabilitaet_je_split']) == {'train', 'val', 'test'}, 'alle Splits im Gesamtlauf'
                assert out['beschreibungen']['einheiten']['kalium_zeilen_andere_einheit'] >= 1
                assert out['lauf']['zeitvertrag_median_s'] > 0 and out['h1']['sensitivitaeten']['d1_gleiche_paare']['beurteilbar']
                import duckdb
                c = duckdb.connect(str(Path(td) / 'priv' / 's3_gesamt.duckdb'), read_only=True)
                assert c.execute("select count(*) from visit where split='test'").fetchone()[0] == len(info['test'])
                bk = c.execute("select count(*) from mess_v2").fetchone()[0]
                assert bk > 50
                c.close()
        finally:
            WAVE = alt_wave

    n_ok = sum(1 for _, ok, _ in ergebnisse if ok)
    print(f'{n_ok} von {len(ergebnisse)} bestanden')
    return n_ok == len(ergebnisse)


# =========================================================================== Hauptprogramm
def main(argv):
    m = argv[1] if len(argv) > 1 else ''
    if m == 'selbsttest':
        sys.exit(0 if selbsttest() else 1)
    if m == 'sperren':
        modus_sperren()
        return
    if m == 'annotation':
        annotation_vorbereiten()
        return
    if m == 'gesamtlauf':
        nw = int(argv[argv.index('--arbeiter') + 1]) if '--arbeiter' in argv else 1
        gesamtlauf(nw=nw)
        return
    print(__doc__)


if __name__ == '__main__':
    main(sys.argv)
