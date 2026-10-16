"""S5-MC-MED v1.6: externe Pruefung der lokalen HF-Zuordnung (Spezifikation SPEZIFIKATION_S5_MCMED_v1_6.md).

GERUEST. Noch nicht auf echten Daten gelaufen. Alle Funktionen sind an Kunstdaten getestet (Modus selbsttest).
Echte Laeufe (extrahieren, erkenner, pilot, bestaetigung) sind gesperrt, bis K die Spezifikation bindet
(spec_lock_v16.json) und, fuer die Bestaetigung, der Pilot eingefroren ist (freeze_v16.json).
Die alten Skripte s5_mcmed_auswertung.py (v1), _v11 bis _v16 bleiben unveraendert erhalten.

Skriptversion v17 (Spezifikation weiterhin v1.6; Dateinamen, Tabellen und Bindung unveraendert): genau eine Korrektur gegenueber v16 (Code-Review Sol zu v16): Besuche ohne
extrahierbare Einheit fuehrten zu executemany(..., []), Rollback und nie gesetzter Erledigt-Marke. Jetzt wird nur bei nichtleeren Zeilen eingefuegt, der Besuch aber auch
ohne Einheiten als erledigt abgeschlossen und im Zaehler 'ohne_einheiten' gefuehrt.

Aenderungen v1.6 gegenueber v1.5 (K-Entscheidung AE-335, Astra-Empfehlung zum Design; vor jeder Ergebnissicht):
  - je Besuch NUR das laengste Einzelsegment nach Zeitmetadaten (Gleichstand: das fruehere); Extraktion, Original, NK-S und alle Kontrollen auf demselben Segment;
  - mindestens 3 vollstaendige Einheiten; Ankerminimum bei D = 0 ist 2; drei Anker als Sensitivitaet ('drei_anker'); alle Rotationsregeln unveraendert;
  - Nenner unter Z2 (Zweig Z3): nur Beschreibung ohne ZB-Urteil (keine Kontrollen, keine Sensitivitaeten).

Aenderung v1.5 gegenueber v1.4 (Code-Review Sol zu v14, genau ein Punkt): Vor jeder Status- und Urteilsbildung wird die VOLLSTAENDIGE Extraktion der jeweiligen
Population verlangt (Train fuer die Bestaetigung, Val fuer den Pilot); sonst Abbruch mit Grund (Zahl offener Besuche).

Aenderungen v1.4 gegenueber v1.3 (Code-Review Sol zu v13, genau vier Restpunkte):
  - Festhash strukturell ueber ast: jeder Modulcode ausser den beiden Erkennerfunktionen und dem Konfigurationsblock wird mitgehasht;
  - urteilsform wird bei der H0-Ausnahme mit aktualisiert (und persistiert);
  - fehlende Basisdatei/-hash der Basis-Spezifikationen (v1.2 und v1.3) wird zwingend zurueckgewiesen (Bindung und Start);
  - Erkennerwahl nur auf derselben vollstaendigen Val-Grundgesamtheit; Teilbestands-Protokolleintraege werden ausgeschlossen.

Aenderungen v1.3 gegenueber v1.2 (Code-Review Sol, Spezifikation v1.3):
  - Kandidatenmodus fuer die Erkennerwahl (erlaubte Aenderungen werden extrahiert und protokolliert, bevor eingefroren wird);
    ALLES ausser den Erkennerfunktionen (r_zacken, hf_fenster) und der Konfiguration ist unveraenderlich gebunden (Festhash), der Helfer s5_mcmed_zaehlung.py ebenfalls.
  - H0-/Z3-gesperrtes B rechnet auch keine B-Sensitivitaeten; nachtraeglich freigegebener Status wird persistiert; Lesefehler bleiben als ungueltige Fenster erhalten;
    ZB2_lage verlangt ZB-1; H0-Aufschluesselung prueft alle drei Teile; H5 verlangt alle fuenf Bedingungen; PK2 mit +-G/2 exakt; PK-B-Abdeckung ohne Monitor korrekt;
    Zeitsemantik-Hinweis in positiven B-Aussagen.

Aenderungen v1.2 gegenueber v1.1 (Astra-Zweitmeinung zu v1.1, K-Entscheidung AE-317 b):
  - S3: Za = Ueberlegenheit (Untergrenze der Differenz > 0); 10 Prozentpunkte nur beschreibend.
  - S4: Nullvergleich fuer ZB-2 Pflicht, keine Ausnahme; ist der bedingte Nullanteil nicht schaetzbar, ist ZB-2 nicht beurteilbar.
  - S5: Kontrollkriterien intervallgestuetzt je Versatz, mindestens 200 auswertbare Besuche je Kontrollbedingung; Repraesentativitaet der Monitorabdeckung berichtet.
  - S6: Wiederfindung je Zwischengitter-Versatz mindestens 50 % des jeweiligen Kontrollnenners.
  - S7: Zulaessigkeitsregeln der Rotation (Raster, Luecken, Stationaritaet, Nahtsprung, verschiedene Rotationen) vorab; ungeeignete Besuche ausgeschlossen.
  - S8: Zufallszahlgenerator (PCG64), Ziehungsreihenfolge und Duplikatregel gebunden; numpy-Version in der Bindung.
  - S9: Erkennerwahl nur nach technischer Guete mit Mindestausbeute, Protokoll, vor jeder B-Schaetzung eingefroren (Pilot vor dem Einfrieren ohne B).
  - Status Z1 nur mit begruendetem rotierbarem Nenner (Besuche und Patienten), sonst explorativ; nachtraegliche Freigaben tragen nie ein bestaetigendes Urteil.

Aenderungen gegenueber v1 (Zweitmeinung Astra, AE-313; Entscheidung AE-311):
  - Zweigsteuerung: B primaer bei G = 15 min; Status bestaetigend (Z1) / explorativ (Z2) / nicht auswertbar (Z3) aus dem Nenner,
    bestimmt VOR jeder Offset-Schaetzung (nenner_b), Pilot ohne Urteil.
  - H0-Sperre inklusive Populationsminimum; B-Urteil nur nach erfuelltem H0 (Ausnahme --b-trotz-h0, dann ohne Urteil);
    H0-Fehlschlaege nach HF-Variabilitaet aufgeschluesselt.
  - Zielaussage ZB (lokal, falsifizierbar) statt der auf dem +-15-Gitter tautologischen H1.
  - B-Positivkontrolle PK-B (Monitor-HF an den echten Chart-Zeitpunkten, Verschiebungen auch zwischen Gitterpunkten),
    PK-G (synthetische Verlaeufe auf der echten Geometrie: konstant, periodisch, Zufallsgang, wahre Versaetze ausserhalb +-15),
    strukturtreue Nullkontrolle NK-S (zyklische Rotation der Einheiten-HF im Besuch, identischer Nenner).
  - Referenzwiederverwendung: Anker-Kennzahlen und Mindestbedingung (mindestens 3 verschiedene Chart-Anker bei D = 0).
  - NK2 schliesst Besuche desselben Patienten aus; NK3 mit Patienten-Bootstrap; Sensitivitaeten im Hauptlauf;
    keine +-60-min-Kennzahlen fuer B; Code-, Kern- und Konfigurationshash beim Start; extr_done an die Erkennerversion gebunden.

Modi (Python: <local path> -I):
  selbsttest                          Kunstdaten-Tests, keine echten Daten, keine Sperre
  sperren                             spec_lock_v16.json schreiben (nur K, einmalig)
  einfrieren --grund TEXT             nach dem Pilot: erlaubte Aenderungen (Erkenner, Qualitaetsgrenzen) festschreiben
  erkenner                            Erkenner-Kriterium (Pilot, val) gegen Monitor-HF
  extrahieren --split val|train [--max-besuche N]
  pilot [--b-trotz-h0]                Auswertung split_random_val
  bestaetigung [--b-trotz-h0]         Auswertung split_random_train (nur nach einfrieren)
Regeln: nie split_*_test.csv; keine Kennungen in Ausgaben; Kleinzellenregel <10; DuckDB 700 MB, 2 Threads;
Einzelprozess unter 1,5 GB (Start nur bei >= 1,5 GB frei); keine Loeschbefehle.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, os, ast, json, time, copy, hashlib, inspect, importlib.util, datetime as dt
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']})
import numpy as np

HIER = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('s5z', HIER / 's5_mcmed_zaehlung.py')
Z = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Z)           # maske, mask_tree, sha, einheiten_zeiten, vollstaendig, Pfade

PRIV = Z.PRIV
AUS = PRIV / 'auswertung_v16'
SPEC_PATH = Path(str(data_path("s5.nachrechnung_spezifikation_s5_mcmed_v1_6_md")))
LOCK = PRIV / 'spec_lock_v16.json'
PROT = PRIV / 'erkenner_protokoll_v16.jsonl'
BASIS_SPEC_PATH = Path(str(data_path("s5.nachrechnung_spezifikation_s5_mcmed_v1_2_md")))
BASIS3_SPEC_PATH = Path(str(data_path("s5.nachrechnung_spezifikation_s5_mcmed_v1_3_md")))
BASIS4_SPEC_PATH = Path(str(data_path("s5.nachrechnung_spezifikation_s5_mcmed_v1_4_md")))
BASIS5_SPEC_PATH = Path(str(data_path("s5.nachrechnung_spezifikation_s5_mcmed_v1_5_md")))
HELFER_PATH = HIER / 's5_mcmed_zaehlung.py'
FREEZE = PRIV / 'freeze_v16.json'

# --------------------------------------------------------------------------- Parameter (Spezifikation v1.1, Abschnitte 5 bis 8)
CFG = {
    'version': 'v1.6',
    'seeds': {'bootstrap': 270100, 'nk1_permutation': 270101, 'nk2_shuffle': 270102, 'nk3_ausduennen': 270103, 'nk4_skala': 270104,
              'nks_rotation': 270105, 'pkg_verlaeufe': 270106, 'nk3_bootstrap': 270107, 'zb_paar_bootstrap': 270108},
    'bootstrap_n': 1000,
    'nk1_n': 20, 'nk2_n': 20, 'nk3_n': 20, 'nks_n': 20,
    'excl_s': 600,
    'einheit_abstand_s': 300,
    'fenster_B_s': 10, 'fenster_A_s': 60,
    'hf_min': 30.0, 'hf_max': 220.0,
    'min_rr_B': 4, 'rr_min_ms': 60000 / 220, 'rr_max_ms': 60000 / 30, 'rr_cv_max': 0.5,
    'abdeckung_A_min_s': 48.0,
    'ref_wert_min': 30.0, 'ref_wert_max': 220.0,
    'A': {'G_min': 60, 'step_min': 2, 'maxref_s': 90, 'breite_max_min': 120, 'minfehler_max': 5.0, 'min_einheiten': 3, 'min_anker_d0': 0},
    'B': {'G_min': 15, 'step_min': 5, 'maxref_s': 1800, 'breite_max_min': 120, 'minfehler_max': 5.0, 'min_einheiten': 3, 'min_anker_d0': 2},
    'nahe_optimal_bpm': 1.0,
    'shifts_min': [0, -10, 10, -30, 30, -60, 60],
    'dichte_min': [1, 5, 15, 30, 60, 120],
    'skala_nk4': 1.15,
    'strata_klein': 30,
    'lage_min': {'A': 2, 'B': 5},               # Lagefenster: |Offset| <= w
    'fenster_min': {'A': [2, 15], 'B': [0, 5, 10]},   # nur Fenster kleiner als das Gitter (A zusaetzlich 60, falls G > 60)
    'zweig': {'z1_min': 5000, 'z1_pat_min': 2500, 'z2_min': 1000, 'z2_pat_min': 500},
    'mon_halb_s': 30.0, 'mon_max_luecke_s': 90.0,
    'pkb': {'shifts_innen_min': [0, 2, -2, 5, -5, 7, -7, 10, -10], 'shifts_zwischen_min': [2, -2, 7, -7],
            'shifts_aussen_min': [20, -20, 30, -30, 45, -45], 'tol_min': 2.5,
            'e0a_min': 0.50, 'e0l_min': 0.50, 'e0b_median_max': 3.0, 'e0b_q90_max': 5.0, 'e0c_max': 0.15, 'min_besuche': 200,
            'pkb_anteil_von_b_hinweis_min': 0.50},
    'pkg': {'verlaeufe': ['zufallsgang', 'konstant', 'periodisch10', 'periodisch15', 'periodisch20', 'periodisch40'],
            'shifts_min': [0, 7, -7, 30], 'rauschen_bpm': 0.5, 'konst_rauschen_bpm': 1.0, 'sigma_walk': 2.0},
    'nks': {'rot_min_s': 3600, 'raster_s': 300, 'luecke_max_s': 900, 'min_rotationen': 5, 'stationaer_max_bpm': 10.0, 'naht_max_bpm': 10.0},
    'zb': {'a_diff_beschreibend': 0.10, 'b_lage_min': 0.75, 'b_null_min_ident': 10},
    'h0': {'ident_min': 0.80, 'innerhalb2_min': 0.95, 'pop_min_max_abs': 2.0},
    'h3': {'ab1_min': 0.80, 'ab60_max': 0.40},
    'h5': {'median_max': 5.0},
    'h6': {'breite_max': 10.0},
    'nk4_max': 0.05, 'nk2_max': 0.05,
    'erkenner': {'innerhalb_bpm': 5.0, 'anteil_min': 0.95, 'ausbeute_min': 0.80},
    'rng': {'bit_generator': 'PCG64', 'api': 'numpy.random.default_rng',
            'nks_ziehung': 'Besuche aufsteigend nach id, Einmalziehung vor den Wiederholungen, ohne Zuruecklegen, min(nks_n, zulaessige) verschiedene Rotationen je Besuch',
            'nk2_partner': 'mit Zuruecklegen, Besuche in Eingabereihenfolge (aufsteigend nach CSN), Klassen in Einfuegereihenfolge',
            'bootstrap': 'Patienten (MRN) aufsteigend, Ziehung mit Zuruecklegen', 'pkg': 'SeedSequence-Liste [seed, besuchsindex, verlaufscode(, 1 fuer Rauschen)]'},
}
EXTR_KEYS = ['excl_s', 'einheit_abstand_s', 'fenster_B_s', 'fenster_A_s', 'hf_min', 'hf_max', 'min_rr_B', 'rr_min_ms', 'rr_max_ms',
             'rr_cv_max', 'abdeckung_A_min_s']
# Aenderungen, die der Pilot nach dem Einfrieren vornehmen darf (Spezifikation 5.1): Qualitaetsgrenzen innerhalb der Spannen
ERKENNER_FUNKTIONEN = ['r_zacken', 'hf_fenster']      # einziger veraenderlicher Code (zusammen mit den Schluesseln aus PILOT_ERLAUBT in der Konfiguration)
PILOT_ERLAUBT = {'rr_cv_max': (0.3, 0.5), 'abdeckung_A_min_s': (45.0, 54.0)}


# =========================================================================== Signal (Erkenner: darf vom Pilot ersetzt werden)
def r_zacken(x, fs):
    """R-Zacken-Erkennung (Bandpass 5-15 Hz, quadrierte Ableitung, gleitende Integration, adaptive Schwelle).
    Rueckgabe: Positionen (Sample-Indizes). Platzhalter-Erkenner; der Pilot darf ihn ersetzen (Spezifikation 5.1)."""
    from scipy.signal import butter, sosfiltfilt, find_peaks
    from scipy.ndimage import maximum_filter1d
    x = np.nan_to_num(np.asarray(x, float), nan=0.0)
    if len(x) < int(2 * fs):
        return np.empty(0, int)
    sos = butter(2, [5.0, 15.0], btype='band', fs=fs, output='sos')
    y = sosfiltfilt(sos, x - np.median(x))
    e = np.gradient(y) ** 2
    w = max(1, int(0.12 * fs))
    m = np.convolve(e, np.ones(w) / w, mode='same')
    env = maximum_filter1d(m, size=max(3, int(8 * fs)))
    pk, _ = find_peaks(m, height=0.25 * env, distance=int(0.25 * fs))
    pk = pk[m[pk] > 0.05 * np.percentile(m, 99.5)] if len(pk) else pk
    r = int(0.06 * fs)
    out = []
    for p in pk:
        a, b = max(0, p - r), min(len(y), p + r + 1)
        out.append(a + int(np.argmax(np.abs(y[a:b]))))
    out = np.unique(np.array(out, int))
    keep = []
    for p in out:
        if not keep or p - keep[-1] >= int(0.25 * fs):
            keep.append(int(p))
    return np.array(keep, int)


def hf_fenster(peaks_t, a, b, cfg=CFG, modus='B'):
    """Herzfrequenz aus R-Zacken-Zeiten in (a,b]. B: zentriertes 10-s-Fenster, A: nachlaufendes 60-s-Fenster.
    Rueckgabe (hf, anzahl_RR) oder (nan, n)."""
    p = peaks_t[(peaks_t > a) & (peaks_t <= b)]
    if len(p) < 2:
        return np.nan, 0
    rr = np.diff(p) * 1000.0
    n = len(rr)
    if modus == 'B' and n < cfg['min_rr_B']:
        return np.nan, n
    if np.any(rr < cfg['rr_min_ms']) or np.any(rr > cfg['rr_max_ms']):
        return np.nan, n
    if np.std(rr) / np.mean(rr) > cfg['rr_cv_max']:
        return np.nan, n
    if modus == 'A' and (p[-1] - p[0]) < cfg['abdeckung_A_min_s']:
        return np.nan, n
    hf = 60000.0 / np.mean(rr)
    if not (cfg['hf_min'] <= hf <= cfg['hf_max']):
        return np.nan, n
    return float(hf), n


def erkenner_version(cfg=None):
    """Version der Extraktion: Quelltext von r_zacken und hf_fenster plus alle Extraktionsparameter. extr_done und units sind daran gebunden."""
    cfg = cfg or CFG
    h = hashlib.sha256()
    h.update(inspect.getsource(r_zacken).encode('utf-8'))
    h.update(inspect.getsource(hf_fenster).encode('utf-8'))
    h.update(json.dumps({k: cfg[k] for k in EXTR_KEYS}, sort_keys=True).encode('utf-8'))
    h.update(Z.sha(HELFER_PATH).encode('utf-8'))             # Helfer (Einheitenzeiten, Hashfunktion): geaenderter Helfer = neue Erkennerversion
    return h.hexdigest()[:16]


# =========================================================================== S5-Kern (aus aw110/rechnen.py, verallgemeinert)
def raster(G_min, step_min):
    return np.arange(-G_min, G_min + step_min / 2.0, step_min).round(6)


def kurven_anker(ts, hf, ref_t, ref_v, lo, hi, D_min, maxref_s):
    """Vollstaendige Fehlerkurven je Einheit samt Ankerindizes. ts: Einheitenzeiten (s), hf: Einheiten-HF, ref: sortierte Referenz.
    Vollstaendig: Ziel t+off in [lo,hi] und naechste Referenz (nur Referenzen in [lo,hi], Gleichstand -> frueher) <= maxref_s fuer JEDEN off.
    Rueckgabe (L, P, times): L Fehlermatrix (n_vollstaendig, len(D)), P Index der naechsten Referenz in times (gleiche Form), times gefilterte Referenzzeiten."""
    D = np.asarray(D_min, float)
    leer = (np.empty((0, len(D))), np.empty((0, len(D)), int), np.empty(0))
    ok = np.isfinite(hf)
    ts, hf = np.asarray(ts)[ok], np.asarray(hf)[ok]
    if len(ts) == 0 or len(ref_t) == 0:
        return leer
    inside = (ref_t >= lo) & (ref_t <= hi)
    times, values = ref_t[inside], ref_v[inside]
    if len(times) == 0:
        return leer
    off = D * 60.0
    L, P = [], []
    for t, h in zip(ts, hf):
        tg = t + off
        k = np.searchsorted(times, tg)
        left = np.clip(k - 1, 0, len(times) - 1)
        right = np.clip(k, 0, len(times) - 1)
        pick = np.where(np.abs(times[left] - tg) <= np.abs(times[right] - tg), left, right)   # Gleichstand: frueher
        dt_ = np.abs(times[pick] - tg)
        if np.all((dt_ <= maxref_s) & (tg >= lo) & (tg <= hi)):
            L.append(np.abs(values[pick] - h))
            P.append(pick)
    if not L:
        return leer
    return np.array(L), np.array(P, int), times


def kurven(ts, hf, ref_t, ref_v, lo, hi, D_min, maxref_s):
    return kurven_anker(ts, hf, ref_t, ref_v, lo, hi, D_min, maxref_s)[0]


def schaetze(L, D_min, regel):
    """Schaetzer wie aw110.estimate. Rueckgabe dict oder None (keine Kurven)."""
    if len(L) == 0:
        return None
    D = np.asarray(D_min, float)
    step = D[1] - D[0]
    loss = np.median(L, axis=0)
    lo = float(np.min(loss))
    near = D[loss <= lo + CFG['nahe_optimal_bpm']]
    opts = D[np.isclose(loss, lo, rtol=0, atol=1e-10)]
    offset = float(sorted(opts, key=lambda x: (abs(x), x))[0])
    breite = float(near[-1] - near[0])
    zusammenhaengend = bool(np.all(np.isclose(np.diff(near), step)))
    ok = bool(len(L) >= regel['min_einheiten'] and breite <= regel['breite_max_min'] and lo <= regel['minfehler_max']
              and near[0] > D[0] and near[-1] < D[-1] and zusammenhaengend)
    return {'loss': loss, 'minimum': lo, 'offset': offset, 'breite': breite, 'ok': ok, 'n': int(len(L)),
            'near_lo': float(near[0]), 'near_hi': float(near[-1]), 'tie_lo': float(opts[0]), 'tie_hi': float(opts[-1]),
            'n_ties': int(len(opts)), 'near_zusammenhaengend': zusammenhaengend}


def anker_kennzahlen(P, times, D):
    """Referenzwiederverwendung: verschiedene Anker bei D = 0 und ueber das Gitter, Referenzwechsel je Einheit, Spannweite der Anker bei D = 0."""
    if len(P) == 0:
        return {'n_anker_d0': 0, 'n_anker_gesamt': 0, 'wechsel_median': 0.0, 'einheiten_ohne_wechsel_anteil': float('nan'), 'spannweite_min': 0.0}
    D = np.asarray(D, float)
    i0 = int(np.argmin(np.abs(D)))
    a0 = np.unique(P[:, i0])
    wechsel = (np.diff(P, axis=1) != 0).sum(axis=1) if P.shape[1] > 1 else np.zeros(len(P), int)
    return {'n_anker_d0': int(len(a0)), 'n_anker_gesamt': int(len(np.unique(P))), 'wechsel_median': float(np.median(wechsel)),
            'einheiten_ohne_wechsel_anteil': float(np.mean(wechsel == 0)),
            'spannweite_min': float((times[a0].max() - times[a0].min()) / 60.0)}


# =========================================================================== Besuchsdaten
class Besuch:
    __slots__ = ('mrn', 'lo', 'hi', 'ts', 'hfA', 'hfB', 'chart_t', 'chart_v', 'mon_t', 'mon_v', 'hrv5', 'id', 'seg')

    def __init__(self, **k):
        for a in self.__slots__:
            setattr(self, a, k.get(a))
        if self.seg is None and self.ts is not None:
            self.seg = np.zeros(len(self.ts), int)


def kopie(b, **felder):
    k = {a: getattr(b, a) for a in Besuch.__slots__}
    k.update(felder)
    return Besuch(**k)


def besuch_variante(b, modus, bez, cfg=CFG):
    """Sensitivitaeten (Spezifikation 7): abstand900 = jede dritte Einheit je Segment; max6 = hoechstens 6 gleichmaessig gewaehlte Einheiten
    (aus den Einheiten mit gueltiger HF der Beziehung); E0 = Besuchsfenster ohne Kuerzung (Einheiten bleiben die mit E = 600 extrahierten)."""
    if modus == 'E0':
        return kopie(b, lo=b.lo - cfg['excl_s'], hi=b.hi + cfg['excl_s'])
    hf = b.hfA if bez == 'A' else b.hfB
    if modus == 'abstand900':
        keep = np.zeros(len(b.ts), bool)
        for s in np.unique(b.seg):
            idx = np.where(b.seg == s)[0]
            keep[idx[::3]] = True
    elif modus == 'max6':
        keep = np.zeros(len(b.ts), bool)
        val = np.where(np.isfinite(hf))[0]
        if len(val) <= 6:
            keep[:] = True
        else:
            keep[val[np.unique(np.round(np.linspace(0, len(val) - 1, 6)).astype(int))]] = True
    else:
        raise ValueError(modus)
    return kopie(b, ts=b.ts[keep], hfA=b.hfA[keep], hfB=b.hfB[keep], seg=b.seg[keep])


def besuche_aus_duckdb(c, split, ev, cfg=CFG, max_besuche=None):
    """Baut Besuche aus dem privaten Bestand (Tabellen visit, num, units_v16). Nur der genannte Split; nur Einheiten der Erkennerversion ev."""
    assert split in ('val', 'train')
    vis = c.execute("SELECT CSN,MRN,r,d FROM visit WHERE split=? AND r IS NOT NULL AND d IS NOT NULL AND CSN IN "
                    "(SELECT DISTINCT CSN FROM units_v16 WHERE ev=?) ORDER BY CSN", [split, ev]).fetchall()
    if max_besuche:
        vis = vis[:max_besuche]
    ids = {v[0] for v in vis}
    refs = {'Chart': {}, 'Monitor': {}}
    for src in ('Chart', 'Monitor'):
        cur = c.execute("SELECT CSN,t,Value FROM num WHERE Measure='HR' AND Source=? AND Value BETWEEN ? AND ? "
                        "AND CSN IN (SELECT CSN FROM visit WHERE split=?) ORDER BY CSN,t",
                        [src, cfg['ref_wert_min'], cfg['ref_wert_max'], split])
        while True:
            rows = cur.fetchmany(500000)
            if not rows:
                break
            for csn, t, v in rows:
                if csn in ids:
                    refs[src].setdefault(csn, []).append((t, v))
    hrv = dict(c.execute("SELECT CSN,median(Value) FROM num WHERE Measure='5min_HRV' AND Source='Monitor' AND CSN IN "
                         "(SELECT CSN FROM visit WHERE split=?) GROUP BY 1", [split]).fetchall())
    un = {}
    for csn, seg, t, a, b in c.execute("SELECT CSN,seg,t,hfA,hfB FROM units_v16 WHERE ev=? AND CSN IN (SELECT CSN FROM visit WHERE split=?) "
                                       "ORDER BY CSN,t", [ev, split]).fetchall():
        un.setdefault(csn, []).append((t, a if a is not None else np.nan, b if b is not None else np.nan, seg))
    out = []
    for csn, mrn, r, d in vis:
        u = np.array(un.get(csn, []), float).reshape(-1, 4)
        rs = {}
        for src in ('Chart', 'Monitor'):
            x = refs[src].get(csn)
            if x:
                x = np.array(x, float)
                tt, inv = np.unique(x[:, 0], return_inverse=True)     # doppelte Zeiten -> Median (wie S5)
                vv = np.array([np.median(x[inv == i, 1]) for i in range(len(tt))]) if len(tt) != len(x) else x[:, 1]
                rs[src] = (tt, vv)
            else:
                rs[src] = (np.empty(0), np.empty(0))
        out.append(Besuch(id=csn, mrn=mrn, lo=r + cfg['excl_s'], hi=d - cfg['excl_s'], ts=u[:, 0], hfA=u[:, 1], hfB=u[:, 2], seg=u[:, 3].astype(int),
                          chart_t=rs['Chart'][0], chart_v=rs['Chart'][1], mon_t=rs['Monitor'][0], mon_v=rs['Monitor'][1], hrv5=hrv.get(csn)))
    return out


# =========================================================================== Auswertung einer Beziehung
def werte_aus(besuche, bez, G_min, cfg=CFG, hf_skala=1.0, ref_override=None, shift_s=0.0, permutiere=None,
              dichte=None, maxref_override=None, hf_override=None, step_override=None, min_anker=None):
    """Schaetzungen je Besuch fuer Beziehung 'A' (Monitor, hfA) oder 'B' (Chart, hfB).
    shift_s: Einheitenzeiten verschieben (PK2). permutiere: rng (NK1, innerhalb Besuch). dichte: (abstand_min, rng) (NK3).
    ref_override: Funktion besuch -> (t, v) (NK2, PK-B, PK-G). hf_override: Funktion besuch -> HF-Array (NK-S, PK-G).
    step_override, min_anker: Sensitivitaeten. Ergebnis je Besuch: 'nenner' = mind. min_einheiten vollstaendige Einheiten UND Ankerbedingung."""
    rg = dict(cfg[bez])
    if step_override:
        rg['step_min'] = step_override
    if min_anker is not None:
        rg['min_anker_d0'] = min_anker
    D = raster(G_min, rg['step_min'])
    res = []
    for b in besuche:
        hf = np.asarray(hf_override(b) if hf_override is not None else (b.hfA if bez == 'A' else b.hfB), float) * hf_skala
        ts = b.ts + shift_s
        if permutiere is not None:
            ok = np.where(np.isfinite(hf))[0]
            hf = hf.copy()
            hf[ok] = hf[ok][permutiere.permutation(len(ok))]
        if ref_override is not None:
            rt, rv = ref_override(b)
        else:
            rt, rv = (b.mon_t, b.mon_v) if bez == 'A' else (b.chart_t, b.chart_v)
        maxref = rg['maxref_s'] if maxref_override is None else maxref_override
        if dichte is not None:
            ab, rng = dichte
            if len(rt):
                grid = rt[0] + rng.uniform(0, ab * 60.0) + np.arange(0, rt[-1] - rt[0] + ab * 60.0, ab * 60.0)
                j = np.searchsorted(rt, grid)
                l = np.clip(j - 1, 0, len(rt) - 1)
                rr_ = np.clip(j, 0, len(rt) - 1)
                nb = np.where(np.abs(rt[l] - grid) <= np.abs(rt[rr_] - grid), l, rr_)
                nb = nb[np.abs(rt[nb] - grid) <= 30.0]
                sel = np.unique(nb)
                rt, rv = rt[sel], rv[sel]
            maxref = max(maxref, ab * 60.0 / 2.0)
        L, P, times = kurven_anker(ts, hf, rt, rv, b.lo, b.hi, D, maxref)
        e = schaetze(L, D, rg)
        if e is None:
            res.append({'n': 0, 'nenner': False, 'ge_min_roh': False, 'ok': False, 'id': b.id, 'mrn': b.mrn})
        else:
            e = dict(e)
            e.update(anker_kennzahlen(P, times, D))
            anker_ok = e['n_anker_d0'] >= rg.get('min_anker_d0', 0)
            e['ge_min_roh'] = bool(e['n'] >= rg['min_einheiten'])
            e['nenner'] = bool(e['ge_min_roh'] and anker_ok)
            e['ok'] = bool(e['ok'] and anker_ok)
            e['hf_sd'] = float(np.nanstd(hf[np.isfinite(hf)])) if np.isfinite(hf).sum() > 1 else np.nan
            e['id'], e['mrn'] = b.id, b.mrn
            e['hrv5'] = b.hrv5
            e['_loss'] = e.pop('loss')
            res.append(e)
    return res, D


def nenner_b(besuche, G_min, cfg=CFG, bez='B'):
    """Nenner und Ankerbedingung OHNE Schaetzer (kein Offset, kein Minimum): Grundlage der Zweigsteuerung. Nutzt nur Zeiten und die Gueltigkeit der HF.
    Fuer B zusaetzlich der ROTIERBARE Nenner (Besuche im Nenner, die die Zulaessigkeitsregeln der Rotation erfuellen) und seine Patientenzahl."""
    rg = cfg[bez]
    D = raster(G_min, rg['step_min'])
    n1 = n3 = nn = 0
    rot = 0
    pat = set()
    gruende = {}
    for b in besuche:
        rt, rv = (b.mon_t, b.mon_v) if bez == 'A' else (b.chart_t, b.chart_v)
        L, P, times = kurven_anker(b.ts, b.hfA if bez == 'A' else b.hfB, rt, rv, b.lo, b.hi, D, rg['maxref_s'])
        n1 += len(L) >= 1
        if len(L) >= rg['min_einheiten']:
            n3 += 1
            if anker_kennzahlen(P, times, D)['n_anker_d0'] >= rg.get('min_anker_d0', 0):
                nn += 1
                if bez == 'B':
                    grund = rotation_zulaessig(b, cfg)[2]
                    if grund == 'ok':
                        rot += 1
                        pat.add(b.mrn)
                    else:
                        gruende[grund] = gruende.get(grund, 0) + 1
    out = {'besuche_gesamt': len(besuche), 'mit_mind_1_vollstaendig': int(n1), 'mit_mind_3_roh': int(n3), 'nenner': int(nn),
           'ausgeschlossen_durch_anker': int(n3 - nn)}
    if bez == 'B':
        out.update({'rotierbarer_nenner': int(rot), 'rotierbare_patienten': int(len(pat)), 'nicht_rotierbar_gruende': gruende})
    return out


def zweig_bestimmen(n_rot, n_pat, stufe, cfg=CFG):
    """Zweig- und Statusregel (Spezifikation 1). Massgeblich ist der ROTIERBARE Vergleichsnenner (Besuche im Nenner, fuer die die strukturtreue Nullkontrolle
    zulaessig ist) und seine Patientenzahl: Z1 (mind. 5.000 Besuche und 2.500 Patienten) bestaetigend, Z2 (mind. 1.000 Besuche und 500 Patienten) explorativ,
    sonst Z3 nicht auswertbar. Pilot: kein Urteil. Ohne begruendeten rotierbaren Nenner ist der Status nie bestaetigend."""
    z = cfg['zweig']
    if stufe != 'bestaetigung':
        return {'zweig': 'Pilot', 'status': 'pilot', 'urteilsform': 'keine', 'rotierbarer_nenner': int(n_rot), 'rotierbare_patienten': int(n_pat)}
    if n_rot >= z['z1_min'] and n_pat >= z['z1_pat_min']:
        return {'zweig': 'Z1', 'status': 'bestaetigend', 'urteilsform': 'bestaetigt oder nicht bestaetigt', 'rotierbarer_nenner': int(n_rot), 'rotierbare_patienten': int(n_pat)}
    if n_rot >= z['z2_min'] and n_pat >= z['z2_pat_min']:
        return {'zweig': 'Z2', 'status': 'explorativ', 'urteilsform': 'Kriterien erfuellt oder verfehlt, ohne Bestaetigungsurteil', 'rotierbarer_nenner': int(n_rot), 'rotierbare_patienten': int(n_pat)}
    return {'zweig': 'Z3', 'status': 'nicht_auswertbar', 'urteilsform': 'keine', 'rotierbarer_nenner': int(n_rot), 'rotierbare_patienten': int(n_pat)}


def anteil_bootstrap(res, flag_nenner, flag_zaehler, seed, n_boot=None):
    """Patienten-Bootstrap (Cluster MRN) fuer den Anteil sum(zaehler)/sum(nenner)."""
    n_boot = n_boot or CFG['bootstrap_n']
    if not len(res):
        return float('nan'), float('nan'), float('nan'), 0, 0
    mrn = np.array([r['mrn'] for r in res])
    nn = np.array([flag_nenner(r) for r in res], float)
    zz = np.array([flag_zaehler(r) for r in res], float)
    u, inv = np.unique(mrn, return_inverse=True)
    sn, sz = np.zeros(len(u)), np.zeros(len(u))
    np.add.at(sn, inv, nn)
    np.add.at(sz, inv, zz)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(u), len(u))
        d = sn[i].sum()
        vals.append(sz[i].sum() / d if d > 0 else np.nan)
    pt = zz.sum() / nn.sum() if nn.sum() > 0 else np.nan
    lo, hi = np.nanquantile(vals, [0.025, 0.975]) if len(vals) and not np.all(np.isnan(vals)) else (np.nan, np.nan)
    return float(pt), float(lo), float(hi), float(zz.sum()), float(nn.sum())


def bootstrap_paar(mrn, n1, z1, n2, z2, seed, n_boot=None):
    """Patienten-Bootstrap fuer zwei Anteile z1/n1 (Daten) und z2/n2 (Kontrolle) auf DERSELBEN Besuchsmenge und ihre Differenz."""
    n_boot = n_boot or CFG['bootstrap_n']
    mrn = np.asarray(mrn)
    if len(mrn) == 0:
        nan = {'wert': float('nan'), 'ci95': [float('nan'), float('nan')]}
        return {'daten': nan, 'kontrolle': dict(nan), 'differenz': dict(nan)}
    u, inv = np.unique(mrn, return_inverse=True)
    sums = []
    for v in (n1, z1, n2, z2):
        s = np.zeros(len(u))
        np.add.at(s, inv, np.asarray(v, float))
        sums.append(s)
    rng = np.random.default_rng(seed)
    vals = np.full((n_boot, 3), np.nan)
    for k in range(n_boot):
        i = rng.integers(0, len(u), len(u))
        a, b, c_, d = [s[i].sum() for s in sums]
        p1 = b / a if a > 0 else np.nan
        p2 = d / c_ if c_ > 0 else np.nan
        vals[k] = (p1, p2, p1 - p2)
    a, b, c_, d = [s.sum() for s in sums]
    pt = (b / a if a > 0 else np.nan, d / c_ if c_ > 0 else np.nan)
    pt = (pt[0], pt[1], pt[0] - pt[1])
    out = {}
    for j, name in enumerate(('daten', 'kontrolle', 'differenz')):
        col = vals[:, j]
        ci = np.nanquantile(col, [0.025, 0.975]) if not np.all(np.isnan(col)) else (np.nan, np.nan)
        out[name] = {'wert': float(pt[j]), 'ci95': [float(ci[0]), float(ci[1])]}
    return out


def zusammenfassung(res, D, cfg=CFG, seed=None, bez='A'):
    """Kennzahlen: Nenner (mind. 3 vollstaendige Einheiten und Ankerbedingung), identifizierbar, Lagefenster (nur Fenster kleiner als das Gitter),
    Zonenbreite, Populationskurve, Referenzwiederverwendung. Keine Fenster, die das Gitter ausfuellen (fuer B kein +-60)."""
    seed = seed if seed is not None else cfg['seeds']['bootstrap']
    ge1 = [r for r in res if r['n'] >= 1]
    roh = [r for r in res if r.get('ge_min_roh')]
    nen = [r for r in res if r.get('nenner')]
    ident = [r for r in nen if r['ok']]
    S = {'besuche_gesamt': len(res), 'mit_mind_1_vollstaendige_einheit': len(ge1), 'mit_mind_3_roh': len(roh), 'nenner': len(nen),
         'ausgeschlossen_durch_anker': len(roh) - len(nen), 'identifizierbar': len(ident)}
    gmax = float(np.max(np.abs(D)))
    if nen:
        p, lo, hi, z, n = anteil_bootstrap(res, lambda r: r.get('nenner', False), lambda r: r.get('nenner', False) and r['ok'], seed)
        S['identifizierbar_anteil'] = {'wert': p, 'ci95': [lo, hi]}
        S['referenz'] = {
            'anker_d0_median': float(np.median([r['n_anker_d0'] for r in nen])),
            'anker_d0_quartile': np.quantile([r['n_anker_d0'] for r in nen], [.25, .75]).tolist(),
            'anker_gesamt_median': float(np.median([r['n_anker_gesamt'] for r in nen])),
            'wechsel_median': float(np.median([r['wechsel_median'] for r in nen])),
            'einheiten_ohne_wechsel_anteil_mittel': float(np.nanmean([r['einheiten_ohne_wechsel_anteil'] for r in nen])),
            'spannweite_min_median': float(np.median([r['spannweite_min'] for r in nen])),
            'spannweite_min_quartile': np.quantile([r['spannweite_min'] for r in nen], [.25, .75]).tolist(),
            'einheiten_je_besuch_median': float(np.median([r['n'] for r in nen]))}
    if ident:
        offs = np.array([r['offset'] for r in ident])
        S['offset_median_min'] = float(np.median(offs))
        S['breite_median_min'] = float(np.median([r['breite'] for r in ident]))
        S['breite_quartile_min'] = np.quantile([r['breite'] for r in ident], [.25, .75]).tolist()
        S['innerhalb_min'] = {}
        wins = [w for w in cfg['fenster_min'][bez] if w < gmax] + ([60] if (bez == 'A' and gmax > 60) else [])
        for w in wins:
            k = int(np.sum(np.abs(offs) <= w))
            S['innerhalb_min'][str(w)] = {'anzahl': k, 'anteil': k / len(ident)}
        w = cfg['lage_min'][bez]
        pl, lo, hi, _, _ = anteil_bootstrap(ident, lambda r: 1, lambda r: abs(r['offset']) <= w, seed + 1)
        S['lage_anteil'] = {'fenster_min': float(w), 'wert': pl, 'ci95': [lo, hi]}
        S['tie_anteil'] = float(np.mean([r['n_ties'] > 1 for r in ident]))
        if bez == 'A':
            S['kriterium_median_innerhalb_15'] = bool(abs(S['offset_median_min']) <= 15)
    cur = [r['_loss'] for r in nen if '_loss' in r]
    if cur:
        pop = np.median(cur, axis=0)
        S['population_minimum_bei_min'] = float(D[int(np.argmin(pop))])
        S['population_kontrast_bpm'] = float(0.5 * (pop[0] + pop[-1]) - np.min(pop))
        S['population_kurve'] = pop.tolist()
        S['population_kurve_offsets_min'] = D.tolist()
    return S


def sensitivitaeten_ties_zonen(res, D, bez='A', cfg=CFG):
    """Argmin- und Zonenvarianten (nur identifizierbare Besuche) bezogen auf das Lagefenster der Beziehung (A: 2, B: 5 min), nicht auf +-60."""
    ident = [r for r in res if r.get('nenner') and r['ok']]
    out = {}
    if not ident:
        return out
    w = float(cfg['lage_min'][bez])

    def stat(vals, mask=None):
        v = np.array(vals)
        if mask is not None:
            v = v[mask]
        return {'besuche': int(len(v)), 'median_min': float(np.median(v)) if len(v) else None,
                'in_lage': int(np.sum(np.abs(v) <= w)), 'anteil_lage': float(np.mean(np.abs(v) <= w)) if len(v) else None}
    out['lagefenster_min'] = w
    out['offset'] = stat([r['offset'] for r in ident])
    out['tie_mitte'] = stat([(r['tie_lo'] + r['tie_hi']) / 2 for r in ident])
    out['tie_ausschluss'] = stat([r['offset'] for r in ident], np.array([r['n_ties'] == 1 for r in ident]))
    out['nah_niedrig'] = stat([r['near_lo'] for r in ident])
    out['nah_hoch'] = stat([r['near_hi'] for r in ident])
    out['nah_mitte'] = stat([(r['near_lo'] + r['near_hi']) / 2 for r in ident])
    out['ganze_zone_in_lage'] = {'anzahl': int(sum(r['near_lo'] >= -w and r['near_hi'] <= w for r in ident)),
                                 'anteil': float(np.mean([r['near_lo'] >= -w and r['near_hi'] <= w for r in ident]))}
    out['zone_ueberlappt_lage'] = {'anzahl': int(sum(r['near_hi'] >= -w and r['near_lo'] <= w for r in ident)),
                                   'anteil': float(np.mean([r['near_hi'] >= -w and r['near_lo'] <= w for r in ident]))}
    return out


def strata(res, cfg=CFG):
    """Einheitenzahl (3, 4-5, >=6), HF-SD-Terzile (Einheiten-HF), Monitor-HRV-Terzile; Gruppen < 30 Besuche zusammengelegt."""
    ge3 = [r for r in res if r.get('nenner')]
    out = {}
    if not ge3:
        return out

    def gruppiere(keyfn):
        g = {}
        for r in ge3:
            g.setdefault(keyfn(r), []).append(r)
        klein = [r for k, v in g.items() if len(v) < cfg['strata_klein'] for r in v]
        g = {k: v for k, v in g.items() if len(v) >= cfg['strata_klein']}
        if klein:
            g['unter_30_zusammengelegt'] = klein
        return {str(k): {'besuche': len(v), 'identifizierbar': int(sum(r['ok'] for r in v)), 'anteil': float(np.mean([r['ok'] for r in v]))} for k, v in g.items()}
    out['einheitenzahl'] = gruppiere(lambda r: '3' if r['n'] == 3 else ('4-5' if r['n'] <= 5 else '>=6'))
    sd = np.array([r['hf_sd'] for r in ge3], float)
    if np.isfinite(sd).sum() >= 3:
        c1, c2 = np.nanquantile(sd, [1 / 3, 2 / 3])
        out['hf_sd_terzil'] = gruppiere(lambda r: 'unten' if r['hf_sd'] <= c1 else ('mitte' if r['hf_sd'] <= c2 else 'oben'))
        out['hf_sd_terzilgrenzen_bpm'] = [float(c1), float(c2)]
    hv = np.array([r['hrv5'] if r['hrv5'] is not None else np.nan for r in ge3], float)
    if np.isfinite(hv).sum() >= 3:
        c1, c2 = np.nanquantile(hv, [1 / 3, 2 / 3])
        out['monitor_hrv5_terzil'] = gruppiere(lambda r: 'fehlt' if r['hrv5'] is None or not np.isfinite(r['hrv5']) else ('unten' if r['hrv5'] <= c1 else ('mitte' if r['hrv5'] <= c2 else 'oben')))
    return out


# =========================================================================== H0: Tor mit Populationsminimum, Aufschluesselung nach HF-Variabilitaet
def h0_variabilitaet(resA, D, cfg=CFG):
    """Sollte 5 (Astra): H0-Kennzahlen je Terzil der HF-Standardabweichung der Einheiten. Bekannter Sollversatz 0 garantiert bei flacher HF keine Identifizierbarkeit.
    Aufschluesselung der Fehlschlaege; entscheidet das Tor nicht."""
    nen = [r for r in resA if r.get('nenner')]
    sd = np.array([r['hf_sd'] for r in nen], float)
    out = {'nenner': len(nen)}
    if np.isfinite(sd).sum() < 3:
        return out
    c1, c2 = np.nanquantile(sd, [1 / 3, 2 / 3])
    gruppen = {'unten': [], 'mitte': [], 'oben': []}
    for r in nen:
        k = 'unten' if r['hf_sd'] <= c1 else ('mitte' if r['hf_sd'] <= c2 else 'oben')
        gruppen[k].append(r)
    out['terzilgrenzen_bpm'] = [float(c1), float(c2)]
    for k, v in gruppen.items():
        idn = [r for r in v if r['ok']]
        out[k] = {'besuche': len(v), 'identifizierbar': len(idn),
                  'anteil': float(len(idn) / len(v)) if v else None,
                  'innerhalb_2_anteil': float(np.mean([abs(r['offset']) <= 2 for r in idn])) if idn else None,
                  'nicht_identifizierbar': len(v) - len(idn)}
    o = out['oben']
    kurven_o = [r['_loss'] for r in gruppen['oben'] if '_loss' in r]
    pm = float(np.asarray(D, float)[int(np.argmin(np.median(kurven_o, axis=0)))]) if kurven_o else None
    o['populationsminimum_bei_min'] = pm
    out['oben_erfuellt_h0'] = bool(o['anteil'] is not None and o['anteil'] >= cfg['h0']['ident_min']
                                   and o['innerhalb_2_anteil'] is not None and o['innerhalb_2_anteil'] >= cfg['h0']['innerhalb2_min']
                                   and pm is not None and abs(pm) <= cfg['h0']['pop_min_max_abs'])         # alle drei H0-Teile im oberen Terzil
    return out


def h0_pruefen(A, cfg=CFG):
    """H0 (Tor): identifizierbarer Anteil >= 80 %, davon >= 95 % innerhalb +-2 min, Populationsminimum bei 0 +-2 min (alle drei Teile)."""
    h = cfg['h0']
    if not A or A.get('identifizierbar_anteil') is None:
        return {'erfuellt': False, 'grund': 'kein Nenner'}
    inn = A.get('innerhalb_min', {}).get('2', {}).get('anteil')
    pm = A.get('population_minimum_bei_min')
    t1 = bool(A['identifizierbar_anteil']['wert'] >= h['ident_min'])
    t2 = bool(inn is not None and inn >= h['innerhalb2_min'])
    t3 = bool(pm is not None and abs(pm) <= h['pop_min_max_abs'])
    return {'erfuellt': bool(t1 and t2 and t3), 'teil_anteil_ident': t1, 'teil_innerhalb_2': t2, 'teil_populationsminimum': t3}


# =========================================================================== Kontrollen
def kontrolle_verschiebung(besuche, bez, G_min, ident_basis, cfg=CFG):
    """PK2 (differenziell): bekannte Verschiebungen der Einheitenzeiten; Erwartung = unverschobene Schaetzung minus Verschiebung; bei A zusaetzlich absolut.
    Kennzahlen nur mit Fenstern kleiner als das Gitter (A: 15, 60; B: 5, 10); S5-Regel nur bei A."""
    step = cfg[bez]['step_min']
    halb = G_min / 2.0                                              # exakt +-G/2 (B: 7,5 min, auch zwischen Gitterpunkten), nicht auf das Raster abgerundet
    norm = lambda x: int(x) if float(x).is_integer() else float(x)
    shifts = sorted({norm(s) for s in cfg['shifts_min'] if abs(s) <= G_min / 2.0} | {norm(halb), norm(-halb)})
    base = {r['id']: r for r in ident_basis if r.get('nenner') and r['ok']}
    wl = [15, 60] if bez == 'A' else [5, 10]
    out = {}
    for s in shifts:
        res, D = werte_aus(besuche, bez, G_min, cfg, shift_s=s * 60.0)
        both = [(base[r['id']], r) for r in res if r.get('nenner') and r['ok'] and r['id'] in base]
        if not both:
            out[str(s)] = {'beide_identifizierbar': 0}
            continue
        diff = np.array([rr['offset'] - (bb['offset'] - s) for bb, rr in both])
        absol = np.array([rr['offset'] - (-s) for bb, rr in both])
        enth = np.mean([(rr['near_lo'] <= bb['offset'] - s <= rr['near_hi']) for bb, rr in both])
        o = {'beide_identifizierbar': len(both), 'verschoben_mind3': int(sum(bool(r.get('nenner')) for r in res)),
             'verschoben_identifizierbar': int(sum(bool(r.get('nenner')) and r['ok'] for r in res)),
             'bias_min': float(np.mean(diff)), 'median_abs_fehler_min': float(np.median(np.abs(diff))),
             'zone_enthaelt_erwartung': float(enth), 'absolut_median_abs_fehler_min': float(np.median(np.abs(absol)))}
        for w in wl:
            o[f'innerhalb_{w}'] = float(np.mean(np.abs(diff) <= w))
        if bez == 'A':
            o['erfolg_s5_regel'] = bool(np.median(np.abs(diff)) <= 15 and np.mean(np.abs(diff) <= 60) >= 0.9)
        out[str(s)] = o
    return out


def kontrolle_permutation(besuche, bez, G_min, cfg=CFG):
    """NK1: n Neuzuordnungen der Einheiten-HF innerhalb des Besuchs (zerstoert die zeitliche Struktur)."""
    rng = np.random.default_rng(cfg['seeds']['nk1_permutation'])
    zahlen, lagen, nenner = [], [], []
    w = cfg['lage_min'][bez]
    for _ in range(cfg['nk1_n']):
        res, D = werte_aus(besuche, bez, G_min, cfg, permutiere=rng)
        nen = [r for r in res if r.get('nenner')]
        ident = [r for r in nen if r['ok']]
        zahlen.append(len(ident))
        nenner.append(len(nen))
        if ident:
            lagen.append(float(np.mean([abs(r['offset']) <= w for r in ident])))
    nn = float(np.mean(nenner)) if nenner else 0.0
    return {'wiederholungen': float(cfg['nk1_n']), 'nenner_mittel': nn, 'identifizierbar_min': int(min(zahlen)), 'identifizierbar_max': int(max(zahlen)),
            'identifizierbar_anteil_max': float(max(zahlen) / nn) if nn else None,
            'lage_anteil_bereich': [min(lagen), max(lagen)] if lagen else None}


def nk2_zuordnung(besuche, bez, rng):
    """Partnerwahl fuer NK2: gleiche Niveauklasse (mittlere Einheiten-HF in 5/min-Klassen) und Punktzahlklasse; der Partner ist IMMER ein anderer Patient (MRN).
    Zuordnung mit Zuruecklegen; ohne zulaessigen Partner entfaellt der Besuch (ohne_partner)."""
    getr = (lambda b: (b.mon_t, b.mon_v)) if bez == 'A' else (lambda b: (b.chart_t, b.chart_v))
    kl = {}
    for i, b in enumerate(besuche):
        hf = (b.hfA if bez == 'A' else b.hfB)
        hf = hf[np.isfinite(hf)]
        t, v = getr(b)
        if len(hf) and len(t) >= 2:
            key = (int(np.mean(hf) // 5), int(min(len(t), 20) // (2 if bez == 'B' else 10)))
            kl.setdefault(key, []).append(i)
    zuord, ohne = {}, 0
    for key, idx in kl.items():
        idx = np.array(idx)
        mr = np.array([besuche[i].mrn for i in idx])
        pick = rng.integers(0, len(idx), len(idx))
        for _ in range(60):
            bad = mr[pick] == mr
            if not bad.any():
                break
            pick[bad] = rng.integers(0, len(idx), int(bad.sum()))
        for a, p in zip(range(len(idx)), pick):
            if mr[p] == mr[a]:
                ohne += 1
                continue
            zuord[besuche[idx[a]].id] = besuche[idx[p]]
    return zuord, ohne


def kontrolle_shuffle(besuche, bez, G_min, cfg=CFG):
    """NK2: Referenzreihe eines anderen Patienten (gleiche Niveau-/Punktzahlklasse), zeitlich so verschoben, dass der erste Referenzpunkt dieselbe Lage zum
    Besuchsfenster hat. Ergebnis: NULLMODELLSPEZIFISCHE Trefferquote (Referenzgeometrie und Nenner weichen von den Daten ab, deshalb mitberichtet)."""
    rng = np.random.default_rng(cfg['seeds']['nk2_shuffle'])
    getr = (lambda b: (b.mon_t, b.mon_v)) if bez == 'A' else (lambda b: (b.chart_t, b.chart_v))
    w = cfg['lage_min'][bez]
    zahlen, lagen, ohnes = [], [], []
    for rep in range(cfg['nk2_n']):
        zuord, ohne = nk2_zuordnung(besuche, bez, rng)
        ohnes.append(ohne)

        def ov(b):
            p = zuord.get(b.id)
            if p is None:
                return np.empty(0), np.empty(0)
            t, v = getr(p)
            t2 = t + (b.lo - p.lo)
            m = (t2 >= b.lo) & (t2 <= b.hi)
            return t2[m], v[m]
        res, D = werte_aus(besuche, bez, G_min, cfg, ref_override=ov)
        nen = [r for r in res if r.get('nenner')]
        ident = [r for r in nen if r['ok']]
        zahlen.append((len(ident), len(nen)))
        if ident:
            lagen.append(float(np.mean([abs(r['offset']) <= w for r in ident])))
    fr = [a / b for a, b in zahlen if b > 0]
    return {'wiederholungen': float(cfg['nk2_n']), 'falsch_identifiziert_anteil_mittel': float(np.mean(fr)) if fr else None,
            'falsch_identifiziert_anteil_min': float(min(fr)) if fr else None, 'falsch_identifiziert_anteil_max': float(max(fr)) if fr else None,
            'nenner_mittel': float(np.mean([b for a, b in zahlen])), 'ohne_partner_mittel': float(np.mean(ohnes)),
            'lage_anteil_bereich': [min(lagen), max(lagen)] if lagen else None,
            'deutung': 'nullmodellspezifische Trefferquote, keine allgemein kalibrierte Falsch-Identifizierungsrate'}


def kontrolle_ausduennen(besuche, G_min, cfg=CFG):
    """NK3 (nur A): Monitorreferenz auf 1..120 min ausgeduennt, Referenzabstandsgrenze max(90 s, Abstand/2). Anteil je Abstand mit Patienten-Bootstrap
    (je Besuch Mittel ueber die Wiederholungen von Nenner- und Identifizierbarkeitsflag; Seed nk3_bootstrap)."""
    rng = np.random.default_rng(cfg['seeds']['nk3_ausduennen'])
    out = {}
    for ab in cfg['dichte_min']:
        nz = np.zeros(len(besuche))
        zz = np.zeros(len(besuche))
        for _ in range(cfg['nk3_n']):
            res, D = werte_aus(besuche, 'A', G_min, cfg, dichte=(ab, rng))
            nz += np.array([bool(r.get('nenner')) for r in res], float)
            zz += np.array([bool(r.get('nenner')) and r['ok'] for r in res], float)
        nz /= cfg['nk3_n']
        zz /= cfg['nk3_n']
        pseudo = [{'mrn': b.mrn, 'n': nz[i], 'z': zz[i]} for i, b in enumerate(besuche)]
        p, lo, hi, _, _ = anteil_bootstrap(pseudo, lambda r: r['n'], lambda r: r['z'], cfg['seeds']['nk3_bootstrap'])
        out[str(ab)] = {'nenner_mittel': float(nz.sum()), 'anteil': {'wert': p, 'ci95': [lo, hi]}}
    return out


def kontrolle_skala(besuche, G_min, cfg=CFG):
    """NK4 (nur A): Wellenform-HF um 15 % gestreckt; wie viele bleiben unter den Kriterien identifizierbar (Niveaufehler als Zeitversatz)?"""
    res, D = werte_aus(besuche, 'A', G_min, cfg, hf_skala=cfg['skala_nk4'])
    return zusammenfassung(res, D, cfg, bez='A')


# --------------------------------------------------------------------------- B-spezifische Kontrollen
def pkb_referenz(b, s_min, cfg=CFG):
    """Pseudo-Chart fuer PK-B: Monitor-HF an den ECHTEN Chart-Zeitpunkten des Besuchs, mit bekanntem Versatz s (Minuten, auch zwischen Gitterpunkten).
    Wert am Zeitstempel c = Monitorwert der Mitte (c - s); der Monitorstempel m steht fuer das Mittel der 60 s davor, seine Mitte ist m - 30 s.
    Punkte ohne Monitorstempel beidseits innerhalb mon_max_luecke_s entfallen (Ausfall). Rueckgabe (t, v, chartpunkte_im_fenster, davon_abgedeckt)."""
    ct = np.asarray(b.chart_t)
    n_im_fenster = int(((ct >= b.lo) & (ct <= b.hi)).sum()) if len(ct) else 0       # Chartpunkte zaehlen unabhaengig von der Monitorverfuegbarkeit
    if len(ct) == 0 or len(b.mon_t) < 2:
        return np.empty(0), np.empty(0), n_im_fenster, 0
    mt, mv = np.asarray(b.mon_t), np.asarray(b.mon_v)
    q = ct - s_min * 60.0 + cfg['mon_halb_s']
    j = np.searchsorted(mt, q, side='right')
    l = np.clip(j - 1, 0, len(mt) - 1)
    r_ = np.clip(j, 0, len(mt) - 1)
    gap = cfg['mon_max_luecke_s']
    exakt = (j >= 1) & (q == mt[l])
    valid = ((j >= 1) & (j < len(mt)) & ((q - mt[l]) <= gap) & ((mt[r_] - q) <= gap)) | exakt
    den = mt[r_] - mt[l]
    w = np.where(den > 0, (q - mt[l]) / np.where(den > 0, den, 1.0), 0.0)
    v = mv[l] * (1 - w) + mv[r_] * w
    inw = (ct >= b.lo) & (ct <= b.hi)
    return ct[valid], v[valid], int(inw.sum()), int((valid & inw).sum())


def _ci_ok(ci, i):
    return bool(ci is not None and len(ci) == 2 and np.isfinite(ci[i]))


def kontrolle_pkb(besuche, G_min, res_daten, cfg=CFG):
    """PK-B (Positivkontrolle unter B-Bedingungen): gleiche Einheiten-HF (10-s-ECG), gleiches Gitter, gleiche Regeln; Referenz = Monitor-HF an den echten
    Chart-Zeitpunkten, um bekannte Versaetze verschoben. Sollwert = Versatz s. Berichtet Abdeckung/Ausfaelle, den Nenner gegenueber B und die Repraesentativitaet.
    Alle Kriterien INTERVALLGESTUETZT JE VERSATZ und nur bei mindestens min_besuche auswertbaren Besuchen JE Kontrollbedingung:
      E0  (Voraussetzung fuer Za): a) bei s = 0 untere 95-%-Grenze des identifizierbaren Anteils >= e0a_min; c) fuer JEDEN aeusseren Versatz (20, 30, 45 min, beide
          Vorzeichen) obere 95-%-Grenze der scheinbar eindeutigen Innenminima <= e0c_max.
      E0l (zusaetzlich fuer Zb): Lagefenster-Wiederfindung bei s = 0 und bei JEDEM Zwischengitter-Versatz (+-2, +-7) mindestens e0l_min des jeweiligen Kontrollnenners;
          gepoolter Fehler der Zwischengitter-Versaetze Median <= e0b_median_max und 90-%-Quantil <= e0b_q90_max."""
    p = cfg['pkb']
    tol = p['tol_min']
    wl = cfg['lage_min']['B']
    seed = cfg['seeds']['bootstrap']
    mind = p['min_besuche']
    chart_w = cov = 0
    for b in besuche:
        _, _, nin, ncov = pkb_referenz(b, 0.0, cfg)
        chart_w += nin
        cov += ncov
    out = {'abdeckung': {'chartpunkte_im_fenster': int(chart_w), 'davon_mit_monitor': int(cov),
                         'anteil': float(cov / chart_w) if chart_w else None}}
    basis_ids = {r['id'] for r in res_daten if r.get('nenner')}
    je_shift, fehler_zw = {}, {}
    for s in sorted(set(p['shifts_innen_min']) | set(p['shifts_aussen_min'])):
        res, D = werte_aus(besuche, 'B', G_min, cfg, ref_override=lambda b, s=s: pkb_referenz(b, s, cfg)[:2])
        nen = [r for r in res if r.get('nenner')]
        ident = [r for r in nen if r['ok']]
        innen = abs(s) <= G_min
        o = {'nenner': len(nen), 'identifizierbar': len(ident), 'innen': bool(innen), 'mindestumfang_erreicht': bool(len(nen) >= mind)}
        if nen:
            pa, lo, hi, _, _ = anteil_bootstrap(res, lambda r: r.get('nenner', False), lambda r: r.get('nenner', False) and r['ok'], seed)
            o['identifizierbar_anteil'] = {'wert': pa, 'ci95': [lo, hi]}
            if innen:
                pw, lo, hi, _, _ = anteil_bootstrap(res, lambda r: r.get('nenner', False),
                                                    lambda r: r.get('nenner', False) and r['ok'] and abs(r['offset'] - s) <= tol, seed)
                o['wiederfindung_anteil'] = {'wert': pw, 'ci95': [lo, hi]}
                pl_, lo, hi, _, _ = anteil_bootstrap(res, lambda r: r.get('nenner', False),
                                                     lambda r: r.get('nenner', False) and r['ok'] and abs(r['offset'] - s) <= wl, seed)
                o['lage_wiederfindung_anteil'] = {'fenster_min': float(wl), 'wert': pl_, 'ci95': [lo, hi]}
        if s in p['shifts_zwischen_min']:
            fehler_zw[s] = [abs(r['offset'] - s) for r in ident]
        if ident and innen:
            err = np.array([abs(r['offset'] - s) for r in ident])
            o['abs_fehler_median_min'] = float(np.median(err))
            o['abs_fehler_q90_min'] = float(np.quantile(err, 0.9))
            o['bias_min'] = float(np.mean([r['offset'] - s for r in ident]))
        je_shift[str(s)] = o
        if s == 0:
            ids0 = {r['id'] for r in nen}
            out['nenner_vergleich'] = {'b_nenner': len(basis_ids), 'pkb_nenner_s0': len(ids0), 'b_auf_pkb_besuchen_nenner': len(basis_ids & ids0)}
            sub = [r for r in res_daten if r['id'] in ids0 and r.get('nenner')]
            if sub:
                pb, lo, hi, _, _ = anteil_bootstrap(sub, lambda r: 1, lambda r: r['ok'], seed)
                out['nenner_vergleich']['b_identifizierbar_anteil_auf_pkb_besuchen'] = {'wert': pb, 'ci95': [lo, hi]}
    out['je_versatz_min'] = je_shift
    # Repraesentativitaet: gilt die Kontrollguete fuer die B-Besuche? (Monitorabdeckung darf keine unbemerkte Uebertragung erzeugen)
    nv = out['nenner_vergleich']
    anteil_b = float(nv['b_auf_pkb_besuchen_nenner'] / nv['b_nenner']) if nv['b_nenner'] else None
    out['kontrollgueltigkeit'] = {'anteil_des_b_nenners_im_pkb_nenner': anteil_b, 'hinweisgrenze': p['pkb_anteil_von_b_hinweis_min'],
                                  'hinweis_teilmenge': bool(anteil_b is None or anteil_b < p['pkb_anteil_von_b_hinweis_min']),
                                  'text': 'Kontrollguete belegt fuer den PK-B-Nenner, nicht automatisch fuer die uebrigen B-Besuche'}
    # E0
    j0 = je_shift.get('0', {})
    a_ci = (j0.get('identifizierbar_anteil') or {}).get('ci95')
    e0a = bool(_ci_ok(a_ci, 0) and a_ci[0] >= p['e0a_min'])
    aussen_je = {}
    for s in p['shifts_aussen_min']:
        sh = je_shift.get(str(s), {})
        ci = (sh.get('identifizierbar_anteil') or {}).get('ci95')
        aussen_je[str(s)] = {'nenner': sh.get('nenner', 0), 'obere_grenze': float(ci[1]) if _ci_ok(ci, 1) else None,
                             'ok': bool(sh.get('mindestumfang_erreicht') and _ci_ok(ci, 1) and ci[1] <= p['e0c_max'])}
    e0c = bool(all(v['ok'] for v in aussen_je.values()))
    umfang = bool(j0.get('mindestumfang_erreicht') and all(je_shift[str(s)].get('mindestumfang_erreicht') for s in p['shifts_aussen_min']))
    erfuellt = bool(umfang and e0a and e0c)
    # E0l
    lw0 = j0.get('lage_wiederfindung_anteil')
    l0 = bool(lw0 is not None and lw0['wert'] >= p['e0l_min'])
    zw_je = {}
    for s in p['shifts_zwischen_min']:
        sh = je_shift.get(str(s), {})
        lw = sh.get('lage_wiederfindung_anteil')
        zw_je[str(s)] = {'nenner': sh.get('nenner', 0), 'wiederfindung_lagefenster': lw['wert'] if lw else None,
                         'untergrenze': float(lw['ci95'][0]) if lw and _ci_ok(lw['ci95'], 0) else None,
                         'ok': bool(sh.get('mindestumfang_erreicht') and lw is not None and lw['wert'] >= p['e0l_min'])}
    zw_ok = bool(all(v['ok'] for v in zw_je.values()))
    err = [e for v in fehler_zw.values() for e in v]
    e0b = bool(len(err) > 0 and np.median(err) <= p['e0b_median_max'] and np.quantile(err, 0.9) <= p['e0b_q90_max'])
    out['e0'] = {'beurteilbar': umfang, 'mindestumfang_je_bedingung': int(mind), 'a_identifizierung_s0_untergrenze': e0a,
                 'c_aussen_je_versatz': aussen_je, 'c_aussen_ok': e0c, 'erfuellt': erfuellt,
                 'l_lage_wiederfindung_s0': l0, 'zwischengitter_je_versatz': zw_je, 'zwischengitter_je_versatz_ok': zw_ok, 'b_zwischengitter': e0b,
                 'zwischengitter_fehler_median_min': float(np.median(err)) if err else None,
                 'zwischengitter_fehler_q90_min': float(np.quantile(err, 0.9)) if err else None,
                 'lage_aufloesbar': bool(erfuellt and l0 and zw_ok and e0b)}
    return out


def rotation_zulaessig(b, cfg=CFG):
    """Zulaessigkeitsregeln der zyklischen Rotation (VORAB, Spezifikation 10.1). Rueckgabe (idx, zulaessige_k, grund); grund 'ok' oder der erste verletzte Test:
      zu_wenige_einheiten  weniger als 2 gueltige Einheiten
      raster               ein Abstand aufeinanderfolgender gueltiger Einheiten ist kein Vielfaches des Einheitenrasters (z. B. Segmentwechsel)
      luecke               groesster Abstand ueber luecke_max_s (mehr als 2 fehlende Rasterpunkte)
      nicht_stationaer     Median der ersten und der zweiten Haelfte der gueltigen HF unterscheiden sich um mehr als stationaer_max_bpm
      naht_sprung          Median der ersten 3 und der letzten 3 gueltigen HF unterscheiden sich um mehr als naht_max_bpm (kuenstlicher Uebergang an der Rotationsnaht)
      zu_wenige_rotationen weniger als min_rotationen zulaessige Rotationen k (jede verlagerte HF um mindestens rot_min_s verschoben; keine Ersatzrotation darunter)
    Ungeeignete Besuche werden ausgeschlossen (Zielpopulation des Nullvergleichs eingeschraenkt)."""
    n = cfg['nks']
    idx = np.where(np.isfinite(b.hfB))[0]
    m = len(idx)
    if m < 2:
        return idx, [], 'zu_wenige_einheiten'
    t = np.asarray(b.ts, float)[idx]
    hf = np.asarray(b.hfB, float)[idx]
    gaps = np.diff(t)
    q = gaps / n['raster_s']
    if np.any(np.abs(q - np.round(q)) * n['raster_s'] > 1.0):
        return idx, [], 'raster'
    if gaps.max() > n['luecke_max_s']:
        return idx, [], 'luecke'
    h = m // 2
    if abs(np.median(hf[:h]) - np.median(hf[h:])) > n['stationaer_max_bpm']:
        return idx, [], 'nicht_stationaer'
    if abs(np.median(hf[:3]) - np.median(hf[-3:])) > n['naht_max_bpm']:
        return idx, [], 'naht_sprung'
    base = np.arange(m)
    ks = [k for k in range(1, m) if np.min(np.abs(t[(base + k) % m] - t)) >= n['rot_min_s']]
    if len(ks) < n['min_rotationen']:
        return idx, [], 'zu_wenige_rotationen'
    return idx, ks, 'ok'


def ziehe_rotationen(rot, info, rng, nks_n):
    """Ziehungsregel (gebunden): Besuche in der uebergebenen Reihenfolge (aufsteigend nach id), Einmalziehung vorab, je Besuch min(nks_n, zulaessige) VERSCHIEDENE
    Rotationen ohne Zuruecklegen (Generator.choice(..., replace=False))."""
    zieh = {}
    for b in rot:
        ks = np.array(info[b.id][1])
        zieh[b.id] = [int(k) for k in rng.choice(ks, size=min(nks_n, len(ks)), replace=False)]
    return zieh


def kontrolle_nks(besuche, G_min, res_daten, cfg=CFG):
    """NK-S (strukturtreue, BEDINGTE Nullkontrolle): Einheiten-HF zyklisch im Besuch rotieren, nur in Besuchen, die die Zulaessigkeitsregeln (rotation_zulaessig)
    erfuellen und im B-Nenner liegen (rotierbarer Vergleichsnenner). Je Besuch werden min(nks_n, zulaessige) VERSCHIEDENE Rotationen ohne Zuruecklegen gezogen
    (Besuche aufsteigend nach id, Einmalziehung vorab, PCG64 mit Seed nks_rotation); der Nullwert eines Besuchs ist der Mittelwert ueber seine gezogenen Rotationen.
    Einheitenzeiten, Gueltigkeit und Chart-Geometrie bleiben; der Nenner ist identisch (geprueft). Vergleich auf derselben Besuchsmenge mit Patienten-Bootstrap."""
    rng = np.random.default_rng(cfg['seeds']['nks_rotation'])
    w = cfg['lage_min']['B']
    daten_all = {r['id']: r for r in res_daten}
    info = {b.id: rotation_zulaessig(b, cfg) for b in besuche}
    gruende = {}
    for b in besuche:
        if daten_all.get(b.id, {}).get('nenner') and info[b.id][2] != 'ok':
            gruende[info[b.id][2]] = gruende.get(info[b.id][2], 0) + 1
    rot = sorted([b for b in besuche if info[b.id][2] == 'ok' and daten_all.get(b.id, {}).get('nenner')], key=lambda b: b.id)
    zieh = ziehe_rotationen(rot, info, rng, cfg['nks_n'])
    pos = {b.id: i for i, b in enumerate(rot)}
    nen_d = np.ones(len(rot))
    z_d = np.array([float(daten_all[b.id]['ok']) for b in rot])
    l_d = np.array([float(daten_all[b.id]['ok'] and abs(daten_all[b.id]['offset']) <= w) for b in rot])
    n_n, z_n, l_n, cnt = np.zeros(len(rot)), np.zeros(len(rot)), np.zeros(len(rot)), np.zeros(len(rot))
    nenner_gleich = True
    R = max((len(v) for v in zieh.values()), default=0)
    for r in range(R):
        sub = [b for b in rot if len(zieh[b.id]) > r]
        gew = {b.id: zieh[b.id][r] for b in sub}

        def hfo(b):
            idx = info[b.id][0]
            hf = np.full(len(b.ts), np.nan)
            m = len(idx)
            hf[idx] = np.asarray(b.hfB)[idx[(np.arange(m) + gew[b.id]) % m]]
            return hf
        res, D = werte_aus(sub, 'B', G_min, cfg, hf_override=hfo)
        for b, rr in zip(sub, res):
            i = pos[b.id]
            nenner_gleich &= bool(rr.get('nenner'))
            ok = bool(rr.get('nenner')) and rr['ok']
            n_n[i] += float(bool(rr.get('nenner')))
            z_n[i] += float(ok)
            l_n[i] += float(ok and abs(rr['offset']) <= w)
            cnt[i] += 1
    cnt_ = np.where(cnt > 0, cnt, 1)
    n_n, z_n, l_n = n_n / cnt_, z_n / cnt_, l_n / cnt_
    mrn = [b.mrn for b in rot]
    sd = cfg['seeds']['zb_paar_bootstrap']
    ident = bootstrap_paar(mrn, nen_d, z_d, n_n, z_n, sd)
    lage = bootstrap_paar(mrn, z_d, l_d, z_n, l_n, sd + 1000)
    return {'besuche_rotierbar': int(len(rot)), 'patienten_rotierbar': int(len(set(mrn))), 'nicht_rotierbar_gruende': gruende,
            'nenner_rotierbar': float(nen_d.sum()), 'nenner_identisch_zu_daten': bool(nenner_gleich), 'null_identifizierbar_mittel': float(z_n.sum()),
            'identifizierbar_anteil': ident, 'lage_anteil_unter_identifizierbaren': lage, 'lagefenster_min': float(w),
            'gezogene_rotationen_je_besuch_min': int(min((len(v) for v in zieh.values()), default=0)),
            'gezogene_rotationen_je_besuch_max': int(max((len(v) for v in zieh.values()), default=0)),
            'zulaessige_rotationen_median': float(np.median([len(info[b.id][1]) for b in rot])) if rot else None,
            'rotation_min_min': float(cfg['nks']['rot_min_s'] / 60.0), 'rng': cfg['rng']['bit_generator'], 'duplikate': 'keine (ohne Zuruecklegen)',
            'deutung': 'bedingte Nullkontrolle: gilt fuer den rotierbaren Vergleichsnenner, nicht fuer ausgeschlossene Besuche'}


def _traj(typ, rng, t0, t1, g):
    if typ == 'zufallsgang':
        tt = np.arange(t0 - 3600.0, t1 + 3600.0 + 60.0, 60.0)
        hf = np.clip(80 + np.cumsum(rng.normal(0, g['sigma_walk'], len(tt))), 45, 160)
        return lambda t: np.interp(t, tt, hf)
    if typ == 'konstant':
        return lambda t: np.full(np.shape(t), 80.0)
    if typ.startswith('periodisch'):
        P = float(typ[len('periodisch'):]) * 60.0
        ph = rng.uniform(0, 2 * np.pi)
        return lambda t: 80.0 + 10.0 * np.sin(2 * np.pi * (np.asarray(t, float) - t0) / P + ph)
    raise ValueError(typ)


def kontrolle_pkg(besuche, G_min, cfg=CFG):
    """PK-G: synthetische Verlaeufe (Zufallsgang, konstant, periodisch 10/15/20/40 min) auf der ECHTEN Geometrie (Einheitenzeiten, gueltige Einheiten, Chart-Zeitpunkte,
    Besuchsfenster) und mit bekannten Versaetzen (0, +-7 zwischen Gitterpunkten, +30 ausserhalb). Der Nenner ist der von B. Sollwert = s (Referenz am Stempel c zeigt
    den Verlauf bei c - s). Zeigt, wie oft Regeln und Gitter bei konstanten/periodischen Verlaeufen scheinbar eindeutige Innenminima erzeugen und wie die Methode
    ohne Erkenner- und Monitorabdeckungsprobleme wiederfindet."""
    g = cfg['pkg']
    tol = cfg['pkb']['tol_min']
    seed = cfg['seeds']['pkg_verlaeufe']
    idx_of = {b.id: i for i, b in enumerate(besuche)}
    out = {}
    for ty_i, typ in enumerate(g['verlaeufe']):
        noise = g['konst_rauschen_bpm'] if typ == 'konstant' else g['rauschen_bpm']
        cache = {}
        for b in besuche:
            i = idx_of[b.id]
            rng = np.random.default_rng([seed, i, ty_i])
            t0 = float(min(b.lo, b.ts.min() if len(b.ts) else b.lo))
            t1 = float(max(b.hi, b.ts.max() if len(b.ts) else b.hi))
            f = _traj(typ, rng, t0, t1, g)
            rng_n = np.random.default_rng([seed, i, ty_i, 1])
            cache[b.id] = (f, rng_n.normal(0, noise, len(b.ts)), rng_n.normal(0, noise, len(b.chart_t)))
        out[typ] = {}
        for s in g['shifts_min']:
            def hfo(b):
                f, nu, nc = cache[b.id]
                return np.where(np.isfinite(b.hfB), f(b.ts) + nu, np.nan)

            def ref(b, s=s):
                f, nu, nc = cache[b.id]
                return b.chart_t, f(b.chart_t - s * 60.0) + nc
            res, D = werte_aus(besuche, 'B', G_min, cfg, hf_override=hfo, ref_override=ref)
            nen = [r for r in res if r.get('nenner')]
            o = {'nenner': len(nen), 'identifizierbar': int(sum(r['ok'] for r in nen))}
            if nen:
                pa, lo, hi, _, _ = anteil_bootstrap(res, lambda r: r.get('nenner', False), lambda r: r.get('nenner', False) and r['ok'], seed)
                o['identifizierbar_anteil'] = {'wert': pa, 'ci95': [lo, hi]}
                if abs(s) <= G_min and typ != 'konstant':
                    pw, lo, hi, _, _ = anteil_bootstrap(res, lambda r: r.get('nenner', False),
                                                        lambda r: r.get('nenner', False) and r['ok'] and abs(r['offset'] - s) <= tol, seed)
                    o['wiederfindung_anteil'] = {'wert': pw, 'ci95': [lo, hi]}
            out[typ][str(s)] = o
    return out


# --------------------------------------------------------------------------- Sensitivitaeten im Hauptlauf
def _kompakt(res, D, bez, seed, cfg=CFG):
    nen = [r for r in res if r.get('nenner')]
    ident = [r for r in nen if r['ok']]
    d = {'besuche_gesamt': len(res), 'nenner': len(nen), 'identifizierbar': len(ident)}
    if nen:
        p, lo, hi, _, _ = anteil_bootstrap(res, lambda r: r.get('nenner', False), lambda r: r.get('nenner', False) and r['ok'], seed)
        d['anteil'] = {'wert': p, 'ci95': [lo, hi]}
    if ident:
        offs = np.array([r['offset'] for r in ident])
        d['offset_median_min'] = float(np.median(offs))
        d['lage_anteil'] = float(np.mean(np.abs(offs) <= cfg['lage_min'][bez]))
    return d


def sensitivitaeten(besuche, cfg=CFG, nur_a=False):
    """Sensitivitaeten der Spezifikation 7 im Hauptlauf: A: G = 30 (Schritt 1), G = 120, Einheitenabstand 900 s, hoechstens 6 Einheiten, E = 0;
    B: Referenzabstand 60 min, Einheitenabstand 900 s, hoechstens 6 Einheiten, E = 0, ohne Ankerbedingung, drei Anker (statt des Minimums 2)."""
    seed = cfg['seeds']['bootstrap']
    out = {'A': {}, 'B': {}}
    GA, GB = cfg['A']['G_min'], cfg['B']['G_min']
    for name, G, st in (('G30_schritt1', 30, 1), ('G120_schritt2', 120, 2)):
        res, D = werte_aus(besuche, 'A', G, cfg, step_override=st)
        out['A'][name] = _kompakt(res, D, 'A', seed, cfg)
    for modus in ('abstand900', 'max6', 'E0'):
        bv = [besuch_variante(b, modus, 'A', cfg) for b in besuche]
        res, D = werte_aus(bv, 'A', GA, cfg)
        out['A'][modus] = _kompakt(res, D, 'A', seed, cfg)
        if nur_a:
            continue
        bv = [besuch_variante(b, modus, 'B', cfg) for b in besuche]
        res, D = werte_aus(bv, 'B', GB, cfg)
        out['B'][modus] = _kompakt(res, D, 'B', seed, cfg)
    if nur_a:
        return {'A': out['A']}
    res, D = werte_aus(besuche, 'B', GB, cfg, maxref_override=3600)
    out['B']['referenzabstand_60min'] = _kompakt(res, D, 'B', seed, cfg)
    res, D = werte_aus(besuche, 'B', GB, cfg, min_anker=0)
    out['B']['ohne_ankerbedingung'] = _kompakt(res, D, 'B', seed, cfg)
    res, D = werte_aus(besuche, 'B', GB, cfg, min_anker=3)
    out['B']['drei_anker'] = _kompakt(res, D, 'B', seed, cfg)
    return out


# =========================================================================== Erkenner-Kriterium (AE-311 Punkt 4, Pilot)
def erkenner_kriterium(besuche, cfg=CFG):
    """Anteil der Fenster, in denen die Wellenform-HF (A, nachlaufend 60 s) um hoechstens 5/min von der Monitor-HF abweicht. Referenz: lineare
    Interpolation der Monitorstempel am Einheitenzeitpunkt (Stempel = Mittel der 60 s davor), beidseits Stempel innerhalb 90 s. Nenner: Fenster mit gueltiger
    Wellenform-HF und verfuegbarem Monitorwert; Ausbeute (gueltig/alle) wird mitberichtet."""
    n_all = n_gueltig = n_ref = n_ok = 0
    for b in besuche:
        mt, mv = np.asarray(b.mon_t), np.asarray(b.mon_v)
        for t, h in zip(b.ts, b.hfA):
            n_all += 1
            if not np.isfinite(h):
                continue
            n_gueltig += 1
            if len(mt) < 2:
                continue
            j = np.searchsorted(mt, t, side='right')
            if j < 1 or j >= len(mt):
                continue
            if t - mt[j - 1] > cfg['mon_max_luecke_s'] or mt[j] - t > cfg['mon_max_luecke_s']:
                continue
            w = (t - mt[j - 1]) / (mt[j] - mt[j - 1])
            ref = mv[j - 1] * (1 - w) + mv[j] * w
            n_ref += 1
            n_ok += abs(h - ref) <= cfg['erkenner']['innerhalb_bpm']
    anteil = float(n_ok / n_ref) if n_ref else None
    ausbeute = float(n_gueltig / n_all) if n_all else None
    ek = cfg['erkenner']
    return {'fenster_gesamt': int(n_all), 'fenster_gueltig': int(n_gueltig), 'ausbeute': ausbeute, 'ausbeute_min': float(ek['ausbeute_min']),
            'ausbeute_erfuellt': bool(ausbeute is not None and ausbeute >= ek['ausbeute_min']),
            'fenster_mit_monitor': int(n_ref), 'innerhalb_bpm': float(ek['innerhalb_bpm']), 'anteil_innerhalb': anteil,
            'anteil_erfuellt': bool(anteil is not None and anteil >= ek['anteil_min']),
            'kriterium_erfuellt': bool(anteil is not None and anteil >= ek['anteil_min'] and ausbeute is not None and ausbeute >= ek['ausbeute_min'])}


def waehle_erkenner_eintrag(eintraege, grundgesamtheit_sha=None):
    """Auswahlregel (nur technische Guete): unter den protokollierten Kandidaten mit erfuelltem Kriterium (Uebereinstimmung >= 95 % UND Ausbeute >= Mindestausbeute)
    der mit dem hoechsten Anteil innerhalb 5/min, bei Gleichstand der hoechsten Ausbeute, dann der fruehesten Protokollzeit. Keine ZB-Groesse geht ein.
    Es zaehlen NUR Eintraege, die auf der VOLLSTAENDIGEN Val-Grundgesamtheit bewertet wurden (vollstaendig = True) und, wenn angegeben, auf genau derselben
    Grundgesamtheit (gleicher Hash der Besuchsmenge); Teilbestands-Eintraege werden ausgeschlossen."""
    ok = [e for e in eintraege if (e.get('kriterium') or {}).get('kriterium_erfuellt') and e.get('vollstaendig') is True
          and (grundgesamtheit_sha is None or (e.get('grundgesamtheit') or {}).get('sha256') == grundgesamtheit_sha)]
    if not ok:
        return None
    ok.sort(key=lambda e: (-e['kriterium']['anteil_innerhalb'], -e['kriterium']['ausbeute'], e.get('utc', '')))
    return ok[0]


def waehle_erkenner(eintraege, grundgesamtheit_sha=None):
    e = waehle_erkenner_eintrag(eintraege, grundgesamtheit_sha)
    return e['ev'] if e else None


def val_grundgesamtheit(c, split='val'):
    """Vollstaendige Grundgesamtheit der Erkennerwahl: alle Val-Besuche mit Wellenform Ableitung II und gueltigem Besuchsfenster (dieselbe Menge wie offene_besuche).
    Rueckgabe: Zahl und Hash der sortierten Besuchsmenge (der Hash enthaelt keine Kennungen im Klartext)."""
    ids = sorted(x[0] for x in c.execute("SELECT DISTINCT h.CSN FROM hd2 h JOIN visit v USING(CSN) WHERE v.split=? AND v.r IS NOT NULL AND v.d IS NOT NULL", [split]).fetchall())
    return {'split': split, 'besuche': len(ids), 'sha256': hashlib.sha256(','.join(str(i) for i in ids).encode('utf-8')).hexdigest()}


def lies_protokoll(prot=None):
    prot = prot or PROT
    if not prot.exists():
        return []
    return [json.loads(z) for z in prot.read_text(encoding='utf-8').splitlines() if z.strip()]


# =========================================================================== Deutungsregel (Spezifikation 4)
def zielaussage_b(Bblock, status, cfg=CFG):
    """Zielaussage ZB (lokal, falsifizierbar), zwei Stufen:
      ZB-1 Zuordnung: E0 (PK-B tragt je Versatz, intervallgestuetzt) und Za = UEBERLEGENHEIT: Untergrenze des 95-%-Intervalls der Differenz des identifizierbaren
           Anteils (Daten minus NK-S auf derselben Besuchsmenge) > 0. Die Differenz von 10 Prozentpunkten und die Untergrenze >= 0,10 werden nur beschreibend gemeldet
           (keine Mindestueberlegenheit behauptet).
      ZB-2 Lage: E0l (Lagefenster und Zwischengitter-Versaetze je Versatz) und Zb = Nullvergleich PFLICHT: Der bedingte Nullanteil (|Offset| <= 5 min unter den
           Identifizierbaren von NK-S) muss schaetzbar sein (im Mittel mindestens b_null_min_ident identifizierbare Nullbesuche und endliches Intervall), die Untergrenze
           des 95-%-Intervalls der Differenz zur Null muss > 0 sein, und der Lageanteil der Daten erreicht die beschreibende Mindestanforderung 0,75. Ist der Nullanteil
           nicht schaetzbar, ist ZB-2 NICHT BEURTEILBAR (keine Ausnahme).
    Ist E0 nicht erfuellt oder nicht beurteilbar, wird ZB nicht beurteilt. Verfehlte Kriterien heissen 'nicht gestuetzt', nicht 'Lage ausserhalb +-5 min'.
    Nachtraegliche Freigaben (status['nachtraeglich']) tragen nie ein bestaetigendes Urteil."""
    z = cfg['zb']
    nks = Bblock.get('nks') or {}
    pkb = Bblock.get('pkb') or {}
    e0 = pkb.get('e0') or {}
    ident = nks.get('identifizierbar_anteil') or {}
    lage = nks.get('lage_anteil_unter_identifizierbaren') or {}
    d = ident.get('differenz') or {}
    ld = lage.get('differenz') or {}
    za = bool(d and _ci_ok(d.get('ci95'), 0) and np.isfinite(d.get('wert', np.nan)) and d['ci95'][0] > 0)
    schaetzbar = bool(nks.get('null_identifizierbar_mittel', 0.0) >= z['b_null_min_ident'] and ld and _ci_ok(ld.get('ci95'), 0) and np.isfinite(ld.get('wert', np.nan)))
    ld_d = lage.get('daten') or {}
    mind_lage = bool(ld_d and np.isfinite(ld_d.get('wert', np.nan)) and ld_d['wert'] >= z['b_lage_min'])
    zb = bool(schaetzbar and ld['ci95'][0] > 0 and mind_lage)
    out = {'E0': bool(e0.get('erfuellt', False)), 'E0_beurteilbar': bool(e0.get('beurteilbar', False)), 'E0_lage_aufloesbar': bool(e0.get('lage_aufloesbar', False)),
           'Za_ueberlegenheit': za, 'Za_differenz_beschreibend': d.get('wert'), 'Za_differenz_ge_010_beschreibend': bool(d and np.isfinite(d.get('wert', np.nan)) and d['wert'] >= z['a_diff_beschreibend']),
           'Za_untergrenze_ge_010_beschreibend': bool(d and _ci_ok(d.get('ci95'), 0) and d['ci95'][0] >= z['a_diff_beschreibend']),
           'Zb_nullanteil_schaetzbar': schaetzbar, 'Zb_mindestanforderung_lage_075': mind_lage, 'Zb_lage': zb,
           'kontrollgueltigkeit_hinweis_teilmenge': bool((pkb.get('kontrollgueltigkeit') or {}).get('hinweis_teilmenge', True))}
    out['ZB1_zuordnung'] = bool(out['E0'] and za)
    out['ZB2_lage'] = bool(out['ZB1_zuordnung'] and out['E0_lage_aufloesbar'] and zb)
    nachtr = bool(status.get('nachtraeglich'))
    if status.get('status') == 'bestaetigend' and not nachtr:
        form = 'bestaetigt'
    else:
        form = 'Kriterien erfuellt (explorativ' + (', nachtraeglich freigegeben' if nachtr else '') + ')'
    if status.get('status') not in ('bestaetigend', 'explorativ'):
        out['urteil'] = 'kein Urteil (' + str(status.get('status')) + ')'
    elif not e0.get('beurteilbar', False):
        out['urteil'] = 'nicht beurteilbar (PK-B ohne ausreichenden Umfang je Kontrollbedingung)'
    elif not e0.get('erfuellt', False):
        out['urteil'] = 'nicht beurteilbar (Kontrollen tragen nicht: Empfindlichkeit oder Spezifitaet je Versatz unzureichend)'
    elif not za:
        out['urteil'] = 'lokale Zuordnung nicht gestuetzt (an dieser Teilmenge; Grenze der Teilmenge beachten)'
    elif not out['E0_lage_aufloesbar']:
        out['urteil'] = 'ZB-1 Zuordnung ' + form + '; Lage bei dieser Referenzgeometrie nicht aufloesbar (PK-B), ZB-2 nicht beurteilt'
    elif not schaetzbar:
        out['urteil'] = 'ZB-1 Zuordnung ' + form + '; ZB-2 nicht beurteilbar (bedingter Nullanteil der Lage nicht schaetzbar)'
    elif zb:
        out['urteil'] = 'ZB-1 Zuordnung und ZB-2 Lage ' + form
    else:
        out['urteil'] = 'ZB-1 Zuordnung ' + form + '; ZB-2 Lage nicht gestuetzt (kein Nachweis der Lage innerhalb +-5 min gegenueber der Nullkontrolle; vereinbar mit Dokumentationsverzug oder Uhrversatz, nicht trennbar)'
    if nachtr:
        out['urteil'] += ' [nach verfehltem H0 nachtraeglich freigegeben]'
    if out['urteil'].startswith('ZB-1') and 'Dokumentationsverzug' not in out['urteil']:          # jede positive B-Aussage traegt den Zeitsemantik-Hinweis
        out['urteil'] += ' [Chart-Zeitsemantik: vereinbar mit Dokumentationsverzug oder Uhrversatz, nicht trennbar]'
    out['zeitsemantik_hinweis'] = bool('Dokumentationsverzug' in out['urteil'])
    return out


def deutung(gesamt, status, stufe='bestaetigung', cfg=CFG):
    """Vorab festgelegte Schwellen (Spezifikation 4); gesamt['A'] und gesamt['B'] sind die Blocke aus modus_auswerten. Nur Etikett, keine Ersatzaussage."""
    out = {}
    A = gesamt.get('A') or {}
    Az = A.get('zusammenfassung') or {}
    out['H0'] = h0_pruefen(Az, cfg)
    out['H0_aufschluesselung_hf_variabilitaet'] = A.get('h0_variabilitaet')
    h0_ok = bool(out['H0']['erfuellt'])
    # H3 Dichte
    nk3 = A.get('nk3')
    if nk3:
        ab = sorted(nk3, key=lambda k: float(k))
        w = [nk3[k]['anteil']['wert'] for k in ab]
        mono = bool(all(w[i] >= w[i + 1] - 1e-12 for i in range(len(w) - 1)))
        out['H3'] = bool(mono and nk3['1']['anteil']['wert'] >= cfg['h3']['ab1_min'] and nk3['60']['anteil']['wert'] <= cfg['h3']['ab60_max'])
    # H4 Null
    if A.get('nk1') and A.get('nk2'):
        n2 = A['nk2'].get('falsch_identifiziert_anteil_mittel')
        out['H4'] = bool((A['nk1'].get('identifizierbar_max') or 0) > 0 and n2 is not None and n2 <= cfg['nk2_max'])
    # H5 Wiederfindung
    pk2 = A.get('pk2_verschiebung')
    if pk2:
        fehlend = [k for k in ('0', '10', '-10', '30', '-30') if k not in pk2 or not pk2[k].get('beide_identifizierbar')]
        if fehlend:
            out['H5'] = None                                  # alle fuenf Bedingungen vorgeschrieben; fehlende = nicht beurteilbar
            out['H5_nicht_beurteilbar_bedingungen'] = fehlend
        else:
            teile = [pk2[k] for k in ('0', '10', '-10', '30', '-30')]
            out['H5'] = bool(all(t.get('erfolg_s5_regel') and t['median_abs_fehler_min'] <= cfg['h5']['median_max']
                                 and t['absolut_median_abs_fehler_min'] <= cfg['h5']['median_max'] for t in teile))
    if Az.get('breite_median_min') is not None:
        out['H6'] = bool(Az['breite_median_min'] <= cfg['h6']['breite_max'])
    if A.get('nk4') and A['nk4'].get('identifizierbar_anteil'):
        out['NK4'] = bool(A['nk4']['identifizierbar_anteil']['wert'] <= cfg['nk4_max'])
    # B
    B = gesamt.get('B')
    out['B_gesperrt_durch_H0'] = bool(stufe == 'bestaetigung' and not h0_ok)
    if B and not B.get('gesperrt') and not B.get('nicht_auswertbar') and status.get('status') in ('bestaetigend', 'explorativ'):
        out['ZB'] = zielaussage_b(B, status, cfg)
        Bz = B.get('zusammenfassung') or {}
        if Bz.get('identifizierbar_anteil'):
            out['H2_B_anteil_10_bis_50_beschreibend'] = bool(0.10 <= Bz['identifizierbar_anteil']['wert'] <= 0.50)
    elif B and B.get('gesperrt'):
        out['ZB'] = {'urteil': 'gesperrt: H0 verfehlt, B nicht interpretieren'}
    elif B and B.get('nicht_auswertbar'):
        out['ZB'] = {'urteil': 'kein ZB-Urteil: Nenner unter Z2, nur Beschreibung'}
    elif stufe == 'pilot':
        out['ZB'] = {'urteil': 'Pilot: kein Urteil'}
    return out


# =========================================================================== Sperre, Bindung, Einfrieren
def cfg_hash(cfg=None):
    return hashlib.sha256(json.dumps(cfg if cfg is not None else CFG, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def _fester_ast(pfad=None):
    """Strukturell bestimmte feste Bestandteile: alle Top-Level-Knoten der Datei AUSSER (i) den Funktionen r_zacken und hf_fenster und (ii) der Zuweisung CFG
    (durch cfg_hash und die Pilot-Spannen geschuetzt). Jeder andere Modulcode, auch unmittelbar hinter den Erkennerfunktionen, bleibt enthalten."""
    tree = ast.parse(Path(pfad or __file__).read_text(encoding='utf-8'))
    fest, erk, cfg_n = [], 0, 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ERKENNER_FUNKTIONEN:
            erk += 1
            continue
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CFG' for t in node.targets):
            cfg_n += 1
            continue
        fest.append(node)
    if erk != len(ERKENNER_FUNKTIONEN) or cfg_n != 1:
        raise SystemExit('FESTHASH: Erkennerfunktionen oder Konfigurationsblock nicht eindeutig gefunden.')
    return fest


def fest_hash(pfad=None):
    """Festhash: SHA-256 der AST-Darstellung (ohne Positionsangaben) aller Top-Level-Knoten ausser den zwei Erkennerfunktionen und dem Konfigurationsblock. Nach der
    Bindung unveraenderlich; nur Erkenner und die erlaubten Qualitaetsgrenzen duerfen sich aendern."""
    h = hashlib.sha256()
    for node in _fester_ast(pfad):
        h.update(ast.dump(node, include_attributes=False).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def helfer_hash():
    return Z.sha(HELFER_PATH)


def code_hash(pfad=None):
    return Z.sha(pfad or __file__)


def spec_hash(spec=None):
    return Z.sha(spec or SPEC_PATH)


def _basis_pfade(basis=None, basis3=None, basis4=None, basis5=None):
    return (('basis_spec_sha256', Path(basis or BASIS_SPEC_PATH), 'v1.2'), ('basis13_spec_sha256', Path(basis3 or BASIS3_SPEC_PATH), 'v1.3'),
            ('basis14_spec_sha256', Path(basis4 or BASIS4_SPEC_PATH), 'v1.4'), ('basis15_spec_sha256', Path(basis5 or BASIS5_SPEC_PATH), 'v1.5'))


def modus_sperren(spec=None, lock=None, cfg=None, basis=None, basis3=None, basis4=None, basis5=None):
    spec, lock, cfg = spec or SPEC_PATH, lock or LOCK, cfg if cfg is not None else CFG
    if lock.exists():
        raise SystemExit('spec_lock_v16.json existiert bereits; nicht ueberschrieben.')
    basen = {}
    for key, pfad, name in _basis_pfade(basis, basis3, basis4, basis5):
        if not pfad.exists():
            raise SystemExit(f'BINDUNG VERWEIGERT: Basis-Spezifikation {name} fehlt ({pfad}); die Bindung ohne Basis ist unzulaessig.')
        basen[key] = Z.sha(pfad)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({'spec_sha256': spec_hash(spec), 'spec': str(spec), **basen,
                                'gesperrt_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                                'code_sha256': code_hash(), 'cfg_sha256': cfg_hash(cfg), 'fest_sha256': fest_hash(), 'helfer_sha256': helfer_hash(), 'cfg': cfg,
                                'erkenner_version': erkenner_version(cfg), 'numpy_version': np.__version__, 'rng': cfg['rng']['bit_generator']},
                               indent=1, default=str), encoding='utf-8')
    print('gesperrt:', spec_hash(spec))


def pruefe_aenderungen(lk, cfg):
    """Erlaubte Pilot-Aenderungen: nur die Schluessel aus PILOT_ERLAUBT innerhalb ihrer Spannen; alle uebrigen Parameter gleich der Bindung."""
    a, b = copy.deepcopy(json.loads(json.dumps(lk['cfg'], default=str))), json.loads(json.dumps(cfg, default=str))
    for k, (lo, hi) in PILOT_ERLAUBT.items():
        if not (lo <= b[k] <= hi):
            raise SystemExit(f'VERWEIGERT: {k} = {b[k]} ausserhalb der Spanne {lo} bis {hi}.')
        a.pop(k, None)
        b.pop(k, None)
    if a != b:
        raise SystemExit('VERWEIGERT: Konfiguration weicht ausserhalb der erlaubten Pilot-Aenderungen ab.')


def pruefe_start(stufe, lock=None, freeze=None, spec=None, cfg=None, basis=None, basis3=None, basis4=None, basis5=None):
    """Prueft beim Start: Spezifikation (und ihre Basis) = gebundener Hash; Festhash (alles ausser Erkenner und Konfiguration) = Bindung; Helfer = Bindung;
    numpy-Version = Bindung. Stufen: 'kandidat' (vor dem Einfrieren: Erkennerkandidat mit nur erlaubten Aenderungen darf extrahiert, geprueft und protokolliert werden),
    'pilot' und 'bestaetigung' (Code- und Konfigurationshash = Bindung bzw. Einfrieren; die Bestaetigung verlangt das Einfrieren)."""
    lock, freeze, spec, cfg = lock or LOCK, freeze or FREEZE, spec or SPEC_PATH, cfg if cfg is not None else CFG
    assert stufe in ('kandidat', 'pilot', 'bestaetigung')
    if not lock.exists():
        raise SystemExit('GESPERRT: spec_lock_v16.json fehlt (K bindet die Spezifikation).')
    lk = json.loads(lock.read_text(encoding='utf-8'))
    if lk.get('spec_sha256') != spec_hash(spec):
        raise SystemExit('GESPERRT: Spezifikation weicht vom gebundenen Hash ab.')
    for key, pfad, name in _basis_pfade(basis, basis3, basis4, basis5):
        if not lk.get(key):
            raise SystemExit(f'GESPERRT: Die Bindung enthaelt keinen Hash der Basis-Spezifikation {name} (unzulaessig).')
        if not pfad.exists():
            raise SystemExit(f'GESPERRT: Basis-Spezifikation {name} fehlt.')
        if Z.sha(pfad) != lk[key]:
            raise SystemExit(f'GESPERRT: Basis-Spezifikation {name} weicht vom gebundenen Hash ab.')
    if fest_hash() != lk.get('fest_sha256'):
        raise SystemExit('GESPERRT: unveraenderlicher Code (alles ausser Erkennerfunktionen und Konfiguration) weicht von der Bindung ab.')
    if helfer_hash() != lk.get('helfer_sha256'):
        raise SystemExit('GESPERRT: Helfer s5_mcmed_zaehlung.py weicht von der Bindung ab.')
    if lk.get('numpy_version') != np.__version__:
        raise SystemExit('GESPERRT: numpy-Version weicht von der Bindung ab (Zufallsstrom nicht reproduzierbar gebunden).')
    if stufe == 'kandidat':
        if freeze.exists():
            raise SystemExit('GESPERRT: nach dem Einfrieren sind keine Erkennerkandidaten mehr zulaessig.')
        pruefe_aenderungen(lk, cfg)
        return {'lock': lk, 'ref': lk, 'quelle': 'Bindung', 'kandidat': True}
    if freeze.exists():
        ref = json.loads(freeze.read_text(encoding='utf-8'))
        quelle = 'Einfrieren'
    else:
        if stufe == 'bestaetigung':
            raise SystemExit('GESPERRT: Bestaetigung verlangt das Einfrieren des Piloten (freeze_v16.json fehlt).')
        ref = lk
        quelle = 'Bindung'
    if code_hash() != ref.get('code_sha256'):
        raise SystemExit(f'GESPERRT: Code weicht vom Hash der {quelle} ab.')
    if cfg_hash(cfg) != ref.get('cfg_sha256'):
        raise SystemExit(f'GESPERRT: Konfiguration weicht vom Hash der {quelle} ab.')
    return {'lock': lk, 'ref': ref, 'quelle': quelle, 'kandidat': False}


def modus_einfrieren(grund, lock=None, freeze=None, spec=None, cfg=None, prot=None, basis=None, basis3=None, c=None, basis4=None, basis5=None):
    """Nach dem Pilot, VOR jeder B-Schaetzung: schreibt Code-, Konfigurations- und Erkennerhash fest. Voraussetzungen: alle Bindungspruefungen der Stufe 'kandidat'
    (Festhash, Helfer, erlaubte Konfigurationsaenderungen) und ein Erkenner-Protokolleintrag, der nach der Auswahlregel (technische Guete, Mindestausbeute) der beste ist und
    GENAU diesen Code- und Konfigurationshash und diese Erkennerversion traegt."""
    lock, freeze = lock or LOCK, freeze or FREEZE
    cfg = cfg if cfg is not None else CFG
    if freeze.exists():
        raise SystemExit('freeze_v16.json existiert bereits; nicht ueberschrieben.')
    pruefe_start('kandidat', lock, freeze, spec, cfg, basis, basis3, basis4, basis5)
    ev = erkenner_version(cfg)
    c = c or Z.verbinden()
    gg = val_grundgesamtheit(c)
    if offene_besuche(c, 'val', ev):
        raise SystemExit('EINFRIEREN VERWEIGERT: die aktuelle Erkennerversion ist nicht auf der vollstaendigen Val-Grundgesamtheit extrahiert.')
    eintraege = lies_protokoll(prot)
    best = waehle_erkenner_eintrag(eintraege, gg['sha256'])
    if best is None:
        raise SystemExit('EINFRIEREN VERWEIGERT: kein auf der vollstaendigen Val-Grundgesamtheit protokollierter Erkennerkandidat erfuellt Uebereinstimmung und Mindestausbeute.')
    if best['ev'] != ev or best.get('code_sha256') != code_hash() or best.get('cfg_sha256') != cfg_hash(cfg):
        raise SystemExit('EINFRIEREN VERWEIGERT: aktuelle Version ist nicht der nach der Auswahlregel beste protokollierte Kandidat (Erkennerversion, Code- oder Konfigurationshash).')
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text(json.dumps({'code_sha256': code_hash(), 'cfg_sha256': cfg_hash(cfg), 'erkenner_version': ev, 'grund': grund,
                                  'eingefroren_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'cfg': cfg,
                                  'erkenner_protokoll_eintraege': len(eintraege), 'gewaehlter_kandidat_utc': best.get('utc')}, indent=1, default=str), encoding='utf-8')
    print('eingefroren, Erkennerversion:', ev)


# =========================================================================== Extraktion (gesperrt; an die Erkennerversion gebunden)
def tabellen_anlegen(c):
    c.execute("CREATE TABLE IF NOT EXISTS units_v16(CSN BIGINT, seg INT, t DOUBLE, hfA DOUBLE, hfB DOUBLE, nA INT, nB INT, ev VARCHAR)")
    c.execute("CREATE TABLE IF NOT EXISTS extr_done_v16(CSN BIGINT, ev VARCHAR)")


def offene_besuche(c, split, ev, max_besuche=None):
    """Besuche ohne abgeschlossene Extraktion DER AKTUELLEN Erkennerversion (Zeilen anderer Versionen bleiben unberuehrt und werden nicht gelesen)."""
    todo = c.execute("SELECT DISTINCT h.CSN FROM hd2 h JOIN visit v USING(CSN) WHERE v.split=? AND v.r IS NOT NULL AND v.d IS NOT NULL "
                     "AND h.CSN NOT IN (SELECT CSN FROM extr_done_v16 WHERE ev=?) ORDER BY h.CSN", [split, ev]).fetchall()
    todo = [x[0] for x in todo]
    return todo[:max_besuche] if max_besuche else todo


def stufe_fuer(split, freeze=None):
    """Startpruefung der Extraktion: train = Bestaetigung (verlangt Einfrieren); val vor dem Einfrieren = Kandidat, danach Pilot."""
    if split == 'train':
        return 'bestaetigung'
    return 'pilot' if (freeze or FREEZE).exists() else 'kandidat'


def laengstes_segment(segs):
    """Segmentwahl (v1.6): je Besuch nur das LAENGSTE Einzelsegment nach Zeitmetadaten (Dauer t1 - t0 aus dem Kopf, nicht die Zahl der Einheiten); bei gleicher Dauer das
    FRUEHERE (kleineres t0, dann kleinere Segmentnummer). segs: Liste von (seg, fs, n, t0, t1). Rueckgabe: Liste mit genau einem Tupel (leer, falls keine Segmente)."""
    if not segs:
        return []
    return [sorted(segs, key=lambda x: (-(x[4] - x[3]), x[3], x[0]))[0]]


def modus_extrahieren(split, max_besuche=None, c=None, pruefen=True):
    import psutil
    assert split in ('val', 'train')
    if pruefen:
        pruefe_start(stufe_fuer(split))
        if psutil.virtual_memory().available < 1.5 * 1024 ** 3:
            raise SystemExit('weniger als 1,5 GB frei: Start abgebrochen')
    import wfdb
    c = c or Z.verbinden()
    tabellen_anlegen(c)
    ev = erkenner_version()
    todo = offene_besuche(c, split, ev, max_besuche)
    cfg = CFG
    lesefehler = 0
    ohne_einheiten = 0
    for i, csn in enumerate(todo):
        r, d = c.execute("SELECT r,d FROM visit WHERE CSN=?", [csn]).fetchone()
        segs = laengstes_segment(c.execute("SELECT seg,fs,n,t0,t1 FROM hd2 WHERE CSN=? ORDER BY seg", [csn]).fetchall())      # nur das laengste Einzelsegment
        rows = []
        for seg, fs, n, t0, t1 in segs:
            ts = Z.einheiten_zeiten(t0, t1, r, d, cfg['einheit_abstand_s'])
            if len(ts) == 0:
                continue
            suf = str(csn)[-3:]
            name = str(Z.WAVE / suf / str(csn) / 'II' / f'{csn}_{seg}')
            for t in ts:
                a = max(0, int((t - cfg['fenster_A_s'] - 2 - t0) * fs))
                b = min(int(n), int((t + cfg['fenster_B_s'] / 2 + 2 - t0) * fs))
                try:
                    rec = wfdb.rdrecord(name, sampfrom=a, sampto=b, physical=True)
                except Exception:
                    # Lesefehler bleiben als UNGUELTIGE Fenster erhalten (Nenner "alle Fenster" der Ausbeute); Marke nA = nB = -1; werden gezaehlt
                    rows.append((csn, seg, float(t), np.nan, np.nan, -1, -1, ev))
                    lesefehler += 1
                    continue
                x = rec.p_signal[:, 0]
                pk = r_zacken(x, fs)
                pt = t0 + (a + pk) / fs
                hfA, nA = hf_fenster(pt, t - cfg['fenster_A_s'], t, cfg, 'A')
                hfB, nB = hf_fenster(pt, t - cfg['fenster_B_s'] / 2, t + cfg['fenster_B_s'] / 2, cfg, 'B')
                rows.append((csn, seg, float(t), hfA, hfB, nA, nB, ev))
        c.execute("BEGIN TRANSACTION")                       # Einheiten und Erledigt-Marke gemeinsam (keine Doppelzeilen nach Abbruch)
        try:
            if rows:                                  # leere Liste wuerde von DuckDB abgelehnt; der Besuch wird trotzdem als erledigt abgeschlossen
                c.executemany("INSERT INTO units_v16 VALUES (?,?,?,?,?,?,?,?)",
                              [tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in row) for row in rows])
            else:
                ohne_einheiten += 1
            c.execute("INSERT INTO extr_done_v16 VALUES (?,?)", [csn, ev])
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
        if i % 200 == 0:
            print(json.dumps({'besuche_fertig': i, 'von': len(todo), 'lesefehler': lesefehler, 'ohne_einheiten': ohne_einheiten}), flush=True)
    return {'ev': ev, 'besuche': len(todo), 'lesefehler': lesefehler, 'ohne_einheiten': ohne_einheiten}


# =========================================================================== Gesamtauswertung (gesperrt)
def modus_erkenner():
    """Technische Guete des aktuellen Erkennerkandidaten auf Val (Uebereinstimmung mit Monitor-HF, Ausbeute inkl. Lesefehler-Fenster); haengt die Messung samt Code-,
    Konfigurations- und Erkennerhash an das Erkenner-Protokoll an (nur Anfuegen, keine Loeschung). Vor dem Einfrieren Stufe 'kandidat', danach 'pilot'.
    Es wird keine B-Groesse berechnet."""
    pruefe_start('pilot' if FREEZE.exists() else 'kandidat')
    c = Z.verbinden()
    ev = erkenner_version()
    besuche = besuche_aus_duckdb(c, 'val', ev)
    if not besuche:
        raise SystemExit('keine Einheiten der aktuellen Erkennerversion fuer val: erst extrahieren.')
    gg = val_grundgesamtheit(c)
    offen = offene_besuche(c, 'val', ev)
    if offen:
        raise SystemExit(f'ERKENNER-PROTOKOLL VERWEIGERT: Teilbestand (noch {len(offen)} von {gg["besuche"]} Val-Besuchen nicht extrahiert); Bewertung nur auf der vollstaendigen Grundgesamtheit.')
    k = erkenner_kriterium(besuche)
    lf = c.execute("SELECT count(*) FROM units_v16 WHERE ev=? AND nA=-1 AND CSN IN (SELECT CSN FROM visit WHERE split='val')", [ev]).fetchone()[0]
    PROT.parent.mkdir(parents=True, exist_ok=True)
    with open(PROT, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'ev': ev, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'kriterium': k, 'lesefehler_fenster': int(lf),
                            'code_sha256': code_hash(), 'cfg_sha256': cfg_hash(), 'helfer_sha256': helfer_hash(),
                            'vollstaendig': True, 'grundgesamtheit': gg,
                            'extraktionsparameter': {q: CFG[q] for q in EXTR_KEYS}}, default=str) + '\n')
    print(json.dumps(Z.mask_tree({'erkenner_version': ev, 'kriterium': k, 'lesefehler_fenster': int(lf)}), indent=1))


def modus_auswerten(split, name, b_trotz_h0=False, c=None, pruefen=True, quelle=None):
    stufe = 'pilot' if split == 'val' else 'bestaetigung'
    if pruefen:
        kette = pruefe_start('kandidat' if (stufe == 'pilot' and not FREEZE.exists()) else stufe)
    else:
        kette = {'quelle': quelle or 'ungeprueft'}
    ev = erkenner_version()
    c = c or Z.verbinden()
    # Vollstaendigkeit der Population VOR jeder Status- und Urteilsbildung (Train fuer die Bestaetigung, Val fuer den Pilot): sonst Abbruch mit Grund
    tabellen_anlegen(c)
    gg = val_grundgesamtheit(c, split)
    offen = offene_besuche(c, split, ev)
    if offen:
        raise SystemExit(f'AUSWERTUNG VERWEIGERT: {split}-Extraktion unvollstaendig ({len(offen)} von {gg["besuche"]} Besuchen der Erkennerversion {ev} offen); '
                         f'Status und Urteil nur auf der vollstaendigen Population.')
    besuche = besuche_aus_duckdb(c, split, ev)
    if not besuche:
        raise SystemExit('keine Einheiten der aktuellen Erkennerversion: erst extrahieren.')
    AUS.mkdir(parents=True, exist_ok=True)
    gesamt = {'spec_sha256': spec_hash() if pruefen else None, 'code_sha256': code_hash(), 'cfg_sha256': cfg_hash(), 'fest_sha256': fest_hash(), 'helfer_sha256': helfer_hash(),
              'erkenner_version': ev, 'bindung_quelle': kette['quelle'], 'split': split, 'stufe': stufe, 'population': {'besuche_mit_wellenform': gg['besuche'], 'offen': 0, 'extraktion_vollstaendig': True},
              'rng': {'bit_generator': CFG['rng']['bit_generator'], 'numpy_version': np.__version__}}
    # 1. Nenner, rotierbarer Nenner und Status VOR jeder Offset-Schaetzung
    G_B = CFG['B']['G_min']
    if G_B != 15:
        raise SystemExit('G fuer Beziehung B ist auf 15 min gebunden (AE-311).')
    nb = nenner_b(besuche, G_B)
    status = zweig_bestimmen(nb['rotierbarer_nenner'], nb['rotierbare_patienten'], stufe)
    status['nenner_detail'] = nb
    status_datei = AUS / f'{name}_status.json'
    status_datei.write_text(json.dumps(Z.mask_tree(status), indent=1), encoding='utf-8')
    gesamt['status'] = status
    # 2. Beziehung A (technische Kontrolle)
    GA = CFG['A']['G_min']
    resA, DA = werte_aus(besuche, 'A', GA)
    A = zusammenfassung(resA, DA, bez='A')
    gesamt['A'] = {'zusammenfassung': A, 'h0_variabilitaet': h0_variabilitaet(resA, DA), 'ties_zonen': sensitivitaeten_ties_zonen(resA, DA, 'A'),
                   'strata': strata(resA), 'pk2_verschiebung': kontrolle_verschiebung(besuche, 'A', GA, resA),
                   'nk1': kontrolle_permutation(besuche, 'A', GA), 'nk2': kontrolle_shuffle(besuche, 'A', GA),
                   'nk3': kontrolle_ausduennen(besuche, GA), 'nk4': kontrolle_skala(besuche, GA)}
    h0_ok = h0_pruefen(A)['erfuellt']
    status_urteil = status
    # 3. Beziehung B (primaer) nach Zweig-, Einfrier- und H0-Regel
    b_gerechnet = False
    if status['zweig'] == 'Z3':
        # Nenner unter Z2: nur Beschreibung ohne ZB-Urteil (keine Kontrollen, keine Sensitivitaeten); bei verfehltem H0 auch keine Beschreibung
        if stufe == 'bestaetigung' and not h0_ok:
            gesamt['B'] = {'nicht_auswertbar': True, 'gesperrt': 'H0', 'nenner': nb}
        else:
            resB, DB = werte_aus(besuche, 'B', G_B)
            gesamt['B'] = {'nicht_auswertbar': True, 'nenner': nb, 'hinweis': 'Nenner unter Z2: nur Beschreibung, kein ZB-Urteil, keine Kontrollen',
                           'beschreibung_ohne_urteil': {'zusammenfassung': zusammenfassung(resB, DB, bez='B'), 'ties_zonen': sensitivitaeten_ties_zonen(resB, DB, 'B'), 'strata': strata(resB)}}
    elif stufe == 'pilot' and kette['quelle'] == 'Bindung':
        gesamt['B'] = {'vor_einfrieren': True, 'nenner': nb,
                       'hinweis': 'Pilot vor dem Einfrieren: keine B-Schaetzung, keine Kontrollen, kein ZB (Erkennerwahl nur nach technischer Guete)'}
    elif stufe == 'bestaetigung' and not h0_ok and not b_trotz_h0:
        gesamt['B'] = {'gesperrt': 'H0', 'nenner': nb}
    else:
        resB, DB = werte_aus(besuche, 'B', G_B)
        B = zusammenfassung(resB, DB, bez='B')
        gesamt['B'] = {'zusammenfassung': B, 'ties_zonen': sensitivitaeten_ties_zonen(resB, DB, 'B'), 'strata': strata(resB),
                       'pk2_verschiebung': kontrolle_verschiebung(besuche, 'B', G_B, resB),
                       'pkb': kontrolle_pkb(besuche, G_B, resB), 'pkg': kontrolle_pkg(besuche, G_B),
                       'nk1': kontrolle_permutation(besuche, 'B', G_B), 'nk2': kontrolle_shuffle(besuche, 'B', G_B),
                       'nks': kontrolle_nks(besuche, G_B, resB)}
        b_gerechnet = True
        if stufe == 'bestaetigung' and not h0_ok:
            gesamt['B']['warnung'] = 'H0 verfehlt, per --b-trotz-h0 nachtraeglich freigegeben: nie bestaetigend, Urteil nur explorativ'
            wirksam = dict(status, status='explorativ' if status['status'] in ('bestaetigend', 'explorativ') else status['status'], nachtraeglich=True,
                           urteilsform='Kriterien erfuellt oder verfehlt, ohne Bestaetigungsurteil (nachtraeglich freigegeben nach verfehltem H0)',
                           ursprung={'zweig': status['zweig'], 'status': status['status'], 'urteilsform': status['urteilsform']})
            status_urteil = wirksam
            gesamt['status'] = wirksam                                          # der WIRKSAME Status wird persistiert (Ausgabe und Statusdatei), der Ursprung steht darin
            status_datei.write_text(json.dumps(Z.mask_tree(wirksam), indent=1), encoding='utf-8')
    # 4. Sensitivitaeten: B-Sensitivitaeten nur, wenn B gerechnet werden darf (gleiche Freigabebedingungen)
    gesamt['sensitivitaeten'] = sensitivitaeten(besuche, nur_a=not b_gerechnet)
    gesamt['deutung'] = deutung(gesamt, status_urteil, stufe)
    p = AUS / f'{name}.json'
    ausgabe = Z.mask_tree(gesamt)
    p.write_text(json.dumps(ausgabe, indent=1, default=str), encoding='utf-8')
    print('geschrieben:', p)
    return gesamt


# =========================================================================== Selbsttest mit Kunstdaten
def kunst_ecg(hf_bpm, dauer_s, fs=500, rauschen=0.05, seed=1, drift=True):
    rng = np.random.default_rng(seed)
    n = int(dauer_s * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    rr = 60.0 / hf_bpm
    schlag = np.arange(0.5, dauer_s, rr)
    for s in schlag:
        x += 1.0 * np.exp(-0.5 * ((t - s) / 0.012) ** 2)
        x -= 0.15 * np.exp(-0.5 * ((t - s + 0.04) / 0.012) ** 2)
        x += 0.25 * np.exp(-0.5 * ((t - s - 0.25) / 0.05) ** 2)
        x += 0.1 * np.exp(-0.5 * ((t - s + 0.16) / 0.02) ** 2)
    if drift:
        x += 0.3 * np.sin(2 * np.pi * 0.2 * t)
    x += rauschen * rng.standard_normal(n)
    return x, schlag


def _kunst_besuche(n_besuche=60, offset_min=0.0, seed=5, chart_n=14, flach=False, mrn_pro=2):
    """Besuche mit bekannter Wahrheit: Referenz = HF-Verlauf des Besuchs, Einheiten-HF = Verlauf zur Zeit t + offset (Soll-Offset = offset).
    Chart-Zeitpunkte zufaellig (gerundet auf Minuten), Monitor minuetlich. flach=True: konstanter Verlauf plus Rauschen (kein Zeitinhalt)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_besuche):
        r = 1.0e9 + i * 1e6
        d = r + 6 * 3600
        lo, hi = r + 600, d - 600
        tt = np.arange(r, d + 1, 60.0)
        hf = 80 + np.cumsum(rng.normal(0, 2.0, len(tt)))
        hf = np.clip(hf, 45, 160)
        if flach:
            hf = np.full(len(tt), 80.0)
        fn = lambda t, tt=tt, hf=hf: np.interp(t, tt, hf)
        mt = tt[(tt >= lo) & (tt <= hi)]
        mon_t, mon_v = mt, fn(mt) + (rng.normal(0, 0.4, len(mt)) if flach else 0.0)
        ct = np.sort(rng.uniform(lo, hi, chart_n)).round(-1)
        ct = np.unique(np.round(ct / 60) * 60)
        ch_t, ch_v = ct, fn(ct) + (rng.normal(0, 0.4, len(ct)) if flach else 0.0)
        ts = np.arange(r + 3600, r + 5 * 3600, 300.0)
        h = fn(ts + offset_min * 60.0) + rng.normal(0, 0.4, len(ts))
        out.append(Besuch(id=i, mrn=i // mrn_pro, lo=lo, hi=hi, ts=ts, hfA=h.copy(), hfB=h.copy(), chart_t=ch_t, chart_v=ch_v,
                          mon_t=mon_t, mon_v=mon_v, hrv5=float(rng.uniform(5, 60)), seg=np.zeros(len(ts), int)))
    return out


def selbsttest():
    import tempfile
    ok = True
    n_pruef = 0

    def check(name, cond, info=''):
        nonlocal ok, n_pruef
        n_pruef += 1
        print(('OK    ' if cond else 'FEHLER'), name, info)
        ok &= bool(cond)

    # 1. Kleinzellenmaske
    check('maske 1..9', all(Z.maske(i) == '<10' for i in range(1, 10)) and Z.maske(0) == 0 and Z.maske(10) == 10)
    check('mask_tree verschachtelt', Z.mask_tree({'a': [3, 12], 'b': {'c': np.int64(5)}, 'd': 0.4}) == {'a': ['<10', 12], 'b': {'c': '<10'}, 'd': 0.4})
    # 2. R-Zacken und HF-Fenster
    for hf0 in (45, 72, 110, 160):
        x, schlag = kunst_ecg(hf0, 80, seed=hf0)
        pk = r_zacken(x, 500)
        pt = pk / 500.0
        hfB, nB = hf_fenster(pt, 35, 45, CFG, 'B')
        hfA, nA = hf_fenster(pt, 10, 70, CFG, 'A')
        check(f'R-Zacken HF {hf0}', abs(hfB - hf0) < 1.5 and abs(hfA - hf0) < 1.0, f'B={hfB:.1f} A={hfA:.1f} Schlaege={len(pk)}/{len(schlag)}')
    # 3. Wiederfindung A (Gitter 60/2)
    for off in (0.0, 20.0, -35.0):
        bes = _kunst_besuche(40, offset_min=off)
        res, D = werte_aus(bes, 'A', CFG['A']['G_min'])
        offs = [r['offset'] for r in res if r['nenner'] and r['ok']]
        check(f'Wiederfindung A offset {off}', len(offs) > 25 and abs(np.median(offs) - off) <= 1.0, f'n_ident={len(offs)} Median={np.median(offs) if offs else None}')
    # 4. Randregel
    bes = _kunst_besuche(30, offset_min=60.0)
    res, D = werte_aus(bes, 'A', 30)
    check('Randpunkt: hoechstens 10 % identifizierbar', sum(r['ok'] for r in res) <= 3, f'ident={sum(r["ok"] for r in res)} von {len(res)}')
    # 5. Vollstaendigkeit B: kleines G
    bes = _kunst_besuche(40, offset_min=0.0, chart_n=20)
    res, D = werte_aus(bes, 'B', 15)
    n3 = sum(r['nenner'] for r in res)
    res2, D2 = werte_aus(bes, 'B', 360)
    check('Vollstaendigkeit B: G=15 > G=360', n3 > sum(r['nenner'] for r in res2), f'G15={n3} G360={sum(r["nenner"] for r in res2)}')
    # 6. Gleichstand, 7. Zusammenhang
    L = np.array([[1, 1, 3, 3, 5]] * 3, float)
    e = schaetze(L, np.array([-10, -5, 0, 5, 10.0]), {'min_einheiten': 3, 'breite_max_min': 120, 'minfehler_max': 5.0})
    check('Tie: kleinster |Offset|, Rand-Kriterium', e['offset'] == -5.0 and e['n_ties'] == 2 and not e['ok'])
    L = np.array([[9, 9, 1, 1.5, 1, 9, 9]] * 3, float)
    e = schaetze(L, np.array([-15, -10, -5, 0, 5, 10, 15.0]), {'min_einheiten': 3, 'breite_max_min': 120, 'minfehler_max': 5.0})
    check('Tie bei +-: kleinerer signierter Offset', e['offset'] == -5.0 and e['n_ties'] == 2 and e['ok'])
    L = np.array([[1, 9, 1, 9, 9, 9, 9]] * 3, float)
    e = schaetze(L, np.array([-15, -10, -5, 0, 5, 10, 15.0]), {'min_einheiten': 3, 'breite_max_min': 120, 'minfehler_max': 5.0})
    check('Zone nicht zusammenhaengend', not e['ok'])
    # 8. PK2 (A) und B ohne +-60-Kennzahlen
    bes = _kunst_besuche(40, offset_min=0.0)
    resA, DA = werte_aus(bes, 'A', 60)
    pk2 = kontrolle_verschiebung(bes, 'A', 60, resA)
    check('PK2 A Verschiebung +10 und +30', pk2['10']['median_abs_fehler_min'] <= 1.0 and pk2['10']['absolut_median_abs_fehler_min'] <= 1.0
          and pk2['30']['beide_identifizierbar'] > 0 and pk2['30']['median_abs_fehler_min'] <= 2.0 and 'innerhalb_60' in pk2['10'])
    # 9. NK1, NK2 (Patientenausschluss), NK3 (Bootstrap), NK4
    base = zusammenfassung(resA, DA, bez='A')
    nk1 = kontrolle_permutation(bes, 'A', 60, dict(CFG, nk1_n=3))
    nk2 = kontrolle_shuffle(bes, 'A', 60, dict(CFG, nk2_n=2))
    check('NK1 << Daten', nk1['identifizierbar_max'] < base['identifizierbar'], f"NK1<= {nk1['identifizierbar_max']} Daten={base['identifizierbar']}")
    check('NK2 Falschrate < 20 %', (nk2['falsch_identifiziert_anteil_mittel'] or 0) < 0.2, str(nk2['falsch_identifiziert_anteil_mittel']))
    rng = np.random.default_rng(1)
    zu, ohne = nk2_zuordnung(bes, 'A', rng)
    check('NK2 schliesst Besuche desselben Patienten aus', len(zu) > 20 and all(p.mrn != next(b.mrn for b in bes if b.id == k) for k, p in zu.items()),
          f'zugeordnet={len(zu)} ohne_partner={ohne}')
    bes_eins = _kunst_besuche(6, mrn_pro=6)          # alle Besuche ein Patient: kein zulaessiger Partner
    zu1, ohne1 = nk2_zuordnung(bes_eins, 'A', np.random.default_rng(2))
    check('NK2: ein einziger Patient -> kein Partner', len(zu1) == 0 and ohne1 == 6, f'{len(zu1)} {ohne1}')
    d3 = kontrolle_ausduennen(bes, 30, dict(CFG, nk3_n=3, dichte_min=[1, 60]))
    check('NK3 mit Bootstrap-Intervall', set(d3) == {'1', '60'} and d3['1']['nenner_mittel'] > 0 and d3['1']['anteil']['ci95'][0] <= d3['1']['anteil']['wert'] <= d3['1']['anteil']['ci95'][1],
          str({k: (round(v['nenner_mittel'], 1), round(v['anteil']['wert'], 3)) for k, v in d3.items()}))
    s4 = kontrolle_skala(bes, 30, CFG)
    check('NK4 Skala 1,15 senkt Identifizierbarkeit', s4['identifizierbar'] < base['identifizierbar'], f"{s4['identifizierbar']} < {base['identifizierbar']}")
    # 10. Bootstrap, Strata, Zonen
    r1 = zusammenfassung(resA, DA, bez='A')['identifizierbar_anteil']
    r2 = zusammenfassung(resA, DA, bez='A')['identifizierbar_anteil']
    check('Bootstrap reproduzierbar', r1 == r2)
    st = strata(resA)
    tz = sensitivitaeten_ties_zonen(resA, DA, 'A')
    check('Strata und Zonen vorhanden', 'einheitenzahl' in st and 'tie_mitte' in tz and 'ganze_zone_in_lage' in tz)

    # ---- B-Geometrie: dichtere Chart-Referenz, 60 Besuche, 2 Besuche je Patient
    bB = _kunst_besuche(60, offset_min=0.0, seed=11, chart_n=45)
    resB, DB = werte_aus(bB, 'B', 15)
    SB = zusammenfassung(resB, DB, bez='B')
    # 11. keine +-60-Kennzahlen fuer B
    txt = json.dumps({'z': SB, 'tz': sensitivitaeten_ties_zonen(resB, DB, 'B'), 'pk2': kontrolle_verschiebung(bB, 'B', 15, resB)}, default=str)
    check('B hat keine +-60-Kennzahlen', 'innerhalb_60' not in txt and '_60' not in txt and '"60"' not in txt and set(SB['innerhalb_min']) <= {'0', '5', '10'},
          str(sorted(SB['innerhalb_min'])))
    check('B identifizierbar auf Kunstdaten (Soll 0)', SB['identifizierbar'] >= 0.4 * SB['nenner'] > 10 and abs(SB.get('offset_median_min', 99)) <= 5, f"{SB['identifizierbar']}/{SB['nenner']}")
    # 12. Anker-Kennzahlen und Mindestbedingung
    check('Referenz-Kennzahlen vorhanden', all(k in SB['referenz'] for k in ('anker_d0_median', 'wechsel_median', 'spannweite_min_median', 'einheiten_ohne_wechsel_anteil_mittel')),
          str({k: round(v, 2) if isinstance(v, float) else v for k, v in SB['referenz'].items() if not isinstance(v, list)}))
    b2 = kopie(bB[0], ts=np.arange(bB[0].lo + 3000, bB[0].lo + 3000 + 7 * 300.0, 300.0), chart_t=np.array([bB[0].lo + 3900.0, bB[0].lo + 4500.0]),
               chart_v=np.array([80.0, 82.0]))
    b2.hfB = np.full(len(b2.ts), 81.0)
    b1 = kopie(b2, chart_t=np.array([bB[0].lo + 3900.0]), chart_v=np.array([80.0]))
    r_a, _ = werte_aus([b2], 'B', 15)
    r_b, _ = werte_aus([b2], 'B', 15, min_anker=3)
    r_c, _ = werte_aus([b1], 'B', 15)
    r_d, _ = werte_aus([b1], 'B', 15, min_anker=0)
    check('Ankerminimum 2 (v1.6): 3 Einheiten mit 2 Ankern im Nenner, mit nur 1 Anker nicht; drei Anker als Sensitivitaet schliesst den 2-Anker-Besuch aus',
          CFG['B']['min_anker_d0'] == 2 and r_a[0]['ge_min_roh'] and r_a[0]['nenner'] and r_a[0]['n_anker_d0'] == 2 and not r_b[0]['nenner']
          and r_c[0]['ge_min_roh'] and not r_c[0]['nenner'] and r_c[0]['n_anker_d0'] == 1 and r_d[0]['nenner'],
          f"2 Anker: nenner={r_a[0]['nenner']}; 3 Anker: {r_b[0]['nenner']}; 1 Anker: {r_c[0]['nenner']}")
    # Segmentwahl: laengstes Einzelsegment nach Zeitmetadaten, Gleichstand frueher
    sg = [(1, 500, 100, 1000.0, 2000.0), (2, 500, 100, 5000.0, 7000.0), (3, 500, 100, 9000.0, 11000.0), (4, 500, 100, 20.0, 1500.0)]
    check('Segmentwahl: laengstes Segment nach Dauer (t1 - t0), bei Gleichstand das fruehere; genau ein Segment; leer -> leer',
          laengstes_segment(sg) == [(2, 500, 100, 5000.0, 7000.0)] and laengstes_segment(sg[::-1]) == [(2, 500, 100, 5000.0, 7000.0)]
          and laengstes_segment([(2, 500, 1, 100.0, 300.0), (1, 500, 1, 400.0, 600.0)]) == [(2, 500, 1, 100.0, 300.0)]
          and laengstes_segment([(2, 500, 1, 100.0, 300.0), (1, 500, 1, 100.0, 300.0)]) == [(1, 500, 1, 100.0, 300.0)]
          and laengstes_segment([(1, 500, 100, 0.0, 50.0)]) == [(1, 500, 100, 0.0, 50.0)] and laengstes_segment([]) == [])
    # 13. Nenner/Status ohne Schaetzer; Zweigregel
    _orig = globals()['schaetze']
    globals()['schaetze'] = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Schaetzer vor Status aufgerufen'))
    try:
        nb = nenner_b(bB, 15)
        nb_ok = True
    except RuntimeError:
        nb_ok = False
    finally:
        globals()['schaetze'] = _orig
    check('Nenner/Status ohne Schaetzer berechenbar, gleich dem Nenner der Auswertung', nb_ok and nb['nenner'] == SB['nenner'] and nb['mit_mind_3_roh'] == SB['mit_mind_3_roh']
          and nb['rotierbarer_nenner'] <= nb['nenner'], str(nb))
    zb_ = lambda n, p, st='bestaetigung': zweig_bestimmen(n, p, st)
    check('Zweigregel Z1/Z2/Z3/Pilot (rotierbarer Nenner und Patienten)', zb_(6000, 3000)['zweig'] == 'Z1' and zb_(6000, 3000)['status'] == 'bestaetigend'
          and zb_(6000, 2000)['zweig'] == 'Z2' and zb_(6000, 2000)['status'] == 'explorativ' and zb_(3900, 2500)['zweig'] == 'Z2' and zb_(5000, 2500)['zweig'] == 'Z1'
          and zb_(4999, 3000)['zweig'] == 'Z2' and zb_(1000, 500)['zweig'] == 'Z2' and zb_(1000, 499)['zweig'] == 'Z3' and zb_(999, 900)['zweig'] == 'Z3'
          and zb_(6000, 3000, 'pilot')['status'] == 'pilot')

    # 14. PK-B: Positivkontrolle unter B-Bedingungen, intervallgestuetzt je Versatz
    cfg_t = dict(CFG, mon_halb_s=0.0, pkb=dict(CFG['pkb'], min_besuche=20))   # Kunstdaten: keine Trailing-Mean-Verschiebung, kleiner Mindestumfang je Bedingung
    pkb = kontrolle_pkb(bB, 15, resB, cfg_t)
    j = pkb['je_versatz_min']
    e0 = pkb['e0']
    zw = [j[str(s)]['abs_fehler_median_min'] for s in (2, -2, 7, -7) if 'abs_fehler_median_min' in j[str(s)]]
    check('PK-B s=0: untere Intervallgrenze der Identifizierung >= 50 %, Lagefenster-Wiederfindung', j['0']['identifizierbar_anteil']['ci95'][0] >= 0.5 and e0['a_identifizierung_s0_untergrenze']
          and j['0']['lage_wiederfindung_anteil']['wert'] >= 0.5 and j['0']['nenner'] >= 20, str(j['0'].get('identifizierbar_anteil')))
    check('PK-B Zwischengitter (+-2, +-7): Fehler und Wiederfindung je Versatz >= 50 %', len(zw) == 4 and max(zw) <= 3.0 and e0['b_zwischengitter'] and e0['zwischengitter_je_versatz_ok']
          and all(v['wiederfindung_lagefenster'] >= 0.5 and v['nenner'] >= 20 for v in e0['zwischengitter_je_versatz'].values()), str({k: round(v['wiederfindung_lagefenster'], 2) for k, v in e0['zwischengitter_je_versatz'].items()}))
    check('PK-B aeussere Versaetze: obere Grenze der Innenminima <= 15 % je Versatz', len(e0['c_aussen_je_versatz']) == 6 and e0['c_aussen_ok']
          and all(v['obere_grenze'] is not None and v['obere_grenze'] <= 0.15 for v in e0['c_aussen_je_versatz'].values()), str({k: round(v['obere_grenze'], 2) for k, v in e0['c_aussen_je_versatz'].items()}))
    check('PK-B E0 und E0l erfuellt, Nenner-Vergleich, Abdeckung und Kontrollgueltigkeit berichtet', e0['erfuellt'] and e0['lage_aufloesbar'] and pkb['abdeckung']['anteil'] is not None
          and 'b_nenner' in pkb['nenner_vergleich'] and 'anteil_des_b_nenners_im_pkb_nenner' in pkb['kontrollgueltigkeit'], str(pkb['kontrollgueltigkeit']))
    pkb_streng = kontrolle_pkb(bB, 15, resB, dict(cfg_t, pkb=dict(cfg_t['pkb'], min_besuche=200)))
    check('PK-B: weniger als 200 Besuche je Kontrollbedingung -> nicht beurteilbar', not pkb_streng['e0']['beurteilbar'] and not pkb_streng['e0']['erfuellt'] and not pkb_streng['e0']['lage_aufloesbar'])
    # PK-B Monitorausfall und Repraesentativitaet
    bAus = [kopie(b, mon_t=b.mon_t[(b.mon_t < b.lo + 3 * 3600) | (b.mon_t > b.lo + 4 * 3600)], mon_v=b.mon_v[(b.mon_t < b.lo + 3 * 3600) | (b.mon_t > b.lo + 4 * 3600)]) for b in bB]
    _, _, nin0, nc0 = pkb_referenz(bB[0], 0.0, cfg_t)
    _, _, nin1, nc1 = pkb_referenz(bAus[0], 0.0, cfg_t)
    check('PK-B Monitorausfall wird als fehlende Abdeckung gezaehlt', nin1 == nin0 and nc1 < nc0 and nc0 == nin0, f'{nc0}/{nin0} -> {nc1}/{nin1}')
    pk2B = kontrolle_verschiebung(bB, 'B', 15, resB)
    check('PK2 B: Verschiebungen 0 und exakt +-G/2 = +-7,5 min (zwischen Gitterpunkten)', {'7.5', '-7.5', '0'} <= set(pk2B) and not any(abs(float(k)) > 7.5 for k in pk2B) and pk2B['7.5']['beide_identifizierbar'] > 0, str(sorted(pk2B)))
    resAB, _ = werte_aus(bB, 'A', 60)
    pk2A_ = kontrolle_verschiebung(bB, 'A', 60, resAB)
    check('PK2 A: +-30 min unveraendert (G/2 ganzzahlig)', '30' in pk2A_ and '-30' in pk2A_ and '30.0' not in pk2A_)
    b_ohne = kopie(bB[0], mon_t=np.empty(0), mon_v=np.empty(0))
    _, _, n_ohne, c_ohne = pkb_referenz(b_ohne, 0.0, cfg_t)
    pk_ohne = kontrolle_pkb([b_ohne] + bB[1:], 15, resB, cfg_t)
    check('PK-B zaehlt Chartpunkte auch bei vollstaendigem Monitorausfall (Abdeckung nicht zu guenstig)', n_ohne == nin0 and n_ohne > 0 and c_ohne == 0
          and pk_ohne['abdeckung']['chartpunkte_im_fenster'] == pkb['abdeckung']['chartpunkte_im_fenster'] and pk_ohne['abdeckung']['davon_mit_monitor'] == pkb['abdeckung']['davon_mit_monitor'] - nc0,
          f"{n_ohne} {c_ohne}")
    m_aus = lambda b: np.ones(len(b.mon_t), bool) if b.id % 2 else (b.mon_t < b.lo + 1800)      # jeder zweite Besuch: Monitor faellt ab 30 min aus
    bAus2 = [kopie(b, mon_t=b.mon_t[m_aus(b)], mon_v=b.mon_v[m_aus(b)]) for b in bB]
    pkb_aus = kontrolle_pkb(bAus2, 15, resB, dict(cfg_t, pkb=dict(cfg_t['pkb'], pkb_anteil_von_b_hinweis_min=0.75)))
    check('PK-B Repraesentativitaet: Ausfall verkleinert den PK-B-Nenner gegenueber B und wird als Hinweis gemeldet',
          pkb_aus['kontrollgueltigkeit']['anteil_des_b_nenners_im_pkb_nenner'] < pkb['kontrollgueltigkeit']['anteil_des_b_nenners_im_pkb_nenner'] and pkb_aus['kontrollgueltigkeit']['hinweis_teilmenge'],
          str((pkb['kontrollgueltigkeit']['anteil_des_b_nenners_im_pkb_nenner'], pkb_aus['kontrollgueltigkeit']['anteil_des_b_nenners_im_pkb_nenner'])))
    # 15. PK-G: konstante/periodische Verlaeufe, Versaetze ausserhalb
    pkg = kontrolle_pkg(bB, 15, dict(CFG, pkg=dict(CFG['pkg'], shifts_min=[0, 7, 30])))
    nen_g = {typ: pkg[typ]['0']['nenner'] for typ in pkg}
    a = lambda typ, s: pkg[typ][str(s)].get('identifizierbar_anteil', {}).get('wert', 0.0)
    w = lambda typ, s: pkg[typ][str(s)].get('wiederfindung_anteil', {}).get('wert', 0.0)
    check('PK-G Nenner identisch zu B auf der echten Geometrie', all(v == SB['nenner'] for v in nen_g.values()), str(nen_g))
    check('PK-G Zufallsgang: Wiederfindung bei 0, wenige Innenminima bei Versatz +30', w('zufallsgang', 0) >= 0.5 and a('zufallsgang', 30) <= 0.15, f"{w('zufallsgang', 0):.2f} {a('zufallsgang', 30):.2f}")
    check('PK-G konstant: keine Identifizierbarkeit', a('konstant', 0) <= 0.05 and a('konstant', 7) <= 0.05, f"{a('konstant', 0):.2f}")
    check('PK-G periodisch 10 und 15 min: mehrdeutig im Gitter (nicht identifizierbar)', a('periodisch10', 0) <= 0.1 and a('periodisch15', 0) <= 0.1, f"{a('periodisch10', 0):.2f} {a('periodisch15', 0):.2f}")
    check('PK-G periodisch 20/40 min: lokal eindeutig (Grenze: globale Mehrdeutigkeit unsichtbar)', a('periodisch20', 0) >= 0.5 and a('periodisch40', 0) >= 0.5, f"{a('periodisch20', 0):.2f} {a('periodisch40', 0):.2f}")
    # 16. Rotation: Zulaessigkeitsregeln (vorab) an Kunstdaten mit Luecken, kurzen Aufzeichnungen, Segmentwechsel, Trend, Naht
    base_b = kopie(bB[0], hfB=80.0 + 0.5 * np.sin(np.arange(len(bB[0].ts))))      # regelmaessig, stationaer, 48 Einheiten
    ok_g = rotation_zulaessig(base_b)
    luecke = np.ones(len(base_b.ts), bool)
    luecke[10:16] = False                                                          # 6 fehlende Einheiten = 1.800 s
    b_gap = kopie(base_b, hfB=np.where(luecke, base_b.hfB, np.nan))
    ts_seg = base_b.ts.copy()
    ts_seg[len(ts_seg) // 2:] += 137.0                                             # Segmentwechsel mit Versatz auf dem Raster
    b_seg = kopie(base_b, ts=ts_seg)
    def mit_laenge(n):
        return kopie(base_b, ts=base_b.ts[:n], hfA=base_b.hfA[:n], hfB=base_b.hfB[:n], seg=base_b.seg[:n])
    b_trend = kopie(base_b, hfB=np.linspace(70, 110, len(base_b.ts)))
    h_seam = np.full(len(base_b.ts), 80.0)
    h_seam[:3] = 70.0
    h_seam[-3:] = 92.0
    b_seam = kopie(base_b, hfB=h_seam)
    b_leer = kopie(base_b, hfB=np.where(np.arange(len(base_b.ts)) == 0, 80.0, np.nan))
    gr = {n: rotation_zulaessig(x)[2] for n, x in (('gap', b_gap), ('seg', b_seg), ('trend', b_trend), ('seam', b_seam), ('leer', b_leer), ('n27', mit_laenge(27)), ('n28', mit_laenge(28)), ('n20', mit_laenge(20)))}
    check('Rotationsregeln: regelmaessiger stationaerer Besuch zulaessig, mindestens 5 verschiedene Rotationen, jede um >= 60 min', ok_g[2] == 'ok' and len(ok_g[1]) >= 5
          and all(min(abs(base_b.ts[ok_g[0]][(np.arange(len(ok_g[0])) + k) % len(ok_g[0])] - base_b.ts[ok_g[0]])) >= 3600 for k in ok_g[1]), str(len(ok_g[1])))
    check('Rotationsregeln: Luecke, Raster/Segmentwechsel, Trend, Nahtsprung, fast leer ausgeschlossen', gr['gap'] == 'luecke' and gr['seg'] == 'raster' and gr['trend'] == 'nicht_stationaer'
          and gr['seam'] == 'naht_sprung' and gr['leer'] == 'zu_wenige_einheiten', str(gr))
    check('Rotationsregeln: kurze Aufzeichnungen (27 und 20 Einheiten) ausgeschlossen, 28 gerade zulaessig; keine Ersatzrotation unter 60 min', gr['n27'] == 'zu_wenige_rotationen' and gr['n20'] == 'zu_wenige_rotationen'
          and gr['n28'] == 'ok' and rotation_zulaessig(mit_laenge(28))[1] == [12, 13, 14, 15, 16], str(rotation_zulaessig(mit_laenge(28))[1]))
    # Ziehung: verschieden, ohne Zuruecklegen, reproduzierbar, begrenzt durch die zulaessigen Rotationen
    r28 = kopie(mit_laenge(28), id=900001)
    info_z = {r28.id: rotation_zulaessig(r28), base_b.id: rotation_zulaessig(base_b)}
    z1 = ziehe_rotationen([r28, base_b], info_z, np.random.default_rng(CFG['seeds']['nks_rotation']), 20)
    z2 = ziehe_rotationen([r28, base_b], info_z, np.random.default_rng(CFG['seeds']['nks_rotation']), 20)
    check('Ziehung der Rotationen: ohne Zuruecklegen, verschieden, reproduzierbar (PCG64), hoechstens die zulaessigen', z1 == z2 and len(z1[r28.id]) == 5 and len(set(z1[r28.id])) == 5
          and len(z1[base_b.id]) == 20 and len(set(z1[base_b.id])) == 20 and set(z1[base_b.id]) <= set(info_z[base_b.id][1]) and CFG['rng']['bit_generator'] == 'PCG64'
          and type(np.random.default_rng(1).bit_generator).__name__ == 'PCG64')
    # NK-S auf gemischten Besuchen: ungeeignete werden ausgeschlossen und nach Grund gezaehlt
    cfg_n = dict(CFG, nks_n=4, nks=dict(CFG['nks'], stationaer_max_bpm=80.0, naht_max_bpm=80.0))
    gemischt = [kopie(b, id=1000 + i) for i, b in enumerate([base_b, b_gap, b_seg, mit_laenge(27), base_b])]
    gemischt = [kopie(b, id=2000 + i, mrn=2000 + i) for i, b in enumerate(gemischt)]
    res_g, _ = werte_aus(gemischt, 'B', 15)
    nks_g = kontrolle_nks(gemischt, 15, res_g, dict(CFG, nks_n=4))
    check('NK-S schliesst ungeeignete Besuche aus und zaehlt die Gruende', nks_g['besuche_rotierbar'] == 2 and nks_g['nicht_rotierbar_gruende'] == {'luecke': 1, 'raster': 1, 'zu_wenige_rotationen': 1}
          and nks_g['nenner_identisch_zu_daten'], str(nks_g['nicht_rotierbar_gruende']))
    # 16b. NK-S: strukturtreue Nullkontrolle mit identischem Nenner auf den Dichtedaten
    nks = kontrolle_nks(bB, 15, resB, cfg_n)
    pd_, pn_ = nks['identifizierbar_anteil']['daten']['wert'], nks['identifizierbar_anteil']['kontrolle']['wert']
    nb_n = nenner_b(bB, 15, cfg_n)
    check('NK-S: Nenner identisch zu den Daten, gleich dem rotierbaren Nenner des Status', nks['nenner_identisch_zu_daten'] and nks['nenner_rotierbar'] > 20 and nb_n['rotierbarer_nenner'] == nks['besuche_rotierbar']
          and nb_n['rotierbare_patienten'] == nks['patienten_rotierbar'], f"rotierbar={nks['besuche_rotierbar']} Patienten={nks['patienten_rotierbar']} Gruende={nks['nicht_rotierbar_gruende']}")
    check('NK-S: Daten >> Null (Differenz mit Intervall > 0)', pd_ - pn_ >= 0.3 and nks['identifizierbar_anteil']['differenz']['ci95'][0] > 0, f'daten={pd_:.2f} null={pn_:.2f}')
    check('NK-S: Ziehungsprotokoll (verschiedene Rotationen, PCG64, keine Duplikate)', nks['gezogene_rotationen_je_besuch_max'] == 4 and nks['rng'] == 'PCG64' and 'keine' in nks['duplikate'])
    # 17. Zielaussage ZB und Deutung (S3 Ueberlegenheit, S4 Nullvergleich Pflicht)
    Bblock = {'nks': nks, 'pkb': pkb, 'zusammenfassung': SB}
    zb = zielaussage_b(Bblock, {'status': 'explorativ'}, CFG)
    check('ZB auf Kunstdaten (explorativ): ZB-1 gestuetzt; ZB-2 nicht beurteilbar, weil der bedingte Nullanteil nicht schaetzbar ist (keine Ausnahme)',
          zb['ZB1_zuordnung'] and not zb['ZB2_lage'] and not zb['Zb_nullanteil_schaetzbar'] and 'ZB-2 nicht beurteilbar' in zb['urteil'] and 'explorativ' in zb['urteil'], zb['urteil'][:90])
    lage_ok = {**nks['lage_anteil_unter_identifizierbaren'], 'daten': {'wert': 0.9, 'ci95': [0.8, 0.95]}, 'differenz': {'wert': 0.2, 'ci95': [0.1, 0.3]}}
    nks_ok = {**nks, 'null_identifizierbar_mittel': 50.0, 'lage_anteil_unter_identifizierbaren': lage_ok}
    zb_ok = zielaussage_b({**Bblock, 'nks': nks_ok}, {'status': 'bestaetigend'}, CFG)
    nks_d = {**nks_ok, 'lage_anteil_unter_identifizierbaren': {**lage_ok, 'differenz': {'wert': 0.05, 'ci95': [-0.02, 0.12]}}}
    nks_l = {**nks_ok, 'lage_anteil_unter_identifizierbaren': {**lage_ok, 'daten': {'wert': 0.5, 'ci95': [0.4, 0.6]}}}
    zb_d = zielaussage_b({**Bblock, 'nks': nks_d}, {'status': 'bestaetigend'}, CFG)
    zb_l = zielaussage_b({**Bblock, 'nks': nks_l}, {'status': 'bestaetigend'}, CFG)
    check('ZB-2: Ueberlegenheit gegen die Null (Untergrenze > 0) plus Lage >= 0,75 noetig', zb_ok['ZB2_lage'] and 'ZB-2 Lage bestaetigt' in zb_ok['urteil'] and not zb_d['ZB2_lage'] and 'nicht gestuetzt' in zb_d['urteil']
          and not zb_l['ZB2_lage'] and zb_d['ZB1_zuordnung'], zb_ok['urteil'][:60])
    nks_z1 = {**nks_ok, 'identifizierbar_anteil': {**nks['identifizierbar_anteil'], 'differenz': {'wert': 0.02, 'ci95': [-0.05, 0.09]}}}
    zb_z1 = zielaussage_b({**Bblock, 'nks': nks_z1}, {'status': 'bestaetigend'}, CFG)
    check('ZB2_lage verlangt ZB-1 (Boolean und Text widersprechen sich nicht)', zb_z1['Zb_lage'] and zb_z1['E0_lage_aufloesbar'] and not zb_z1['ZB1_zuordnung'] and not zb_z1['ZB2_lage'] and 'nicht gestuetzt' in zb_z1['urteil'])
    check('Zeitsemantik-Hinweis in positiven ZB-Aussagen (ZB-1 und ZB-1+ZB-2), nicht in Pilot/nicht beurteilbar', 'Dokumentationsverzug' in zb_ok['urteil'] and 'Dokumentationsverzug' in zb['urteil'] and zb_ok['zeitsemantik_hinweis']
          and 'Dokumentationsverzug' not in zielaussage_b(Bblock, {'status': 'pilot'}, CFG)['urteil'])
    zb_a = zielaussage_b({**Bblock, 'nks': nks_ok, 'pkb': {'e0': {**pkb['e0'], 'lage_aufloesbar': False}}}, {'status': 'bestaetigend'}, CFG)
    check('ZB: Lage nicht aufloesbar -> nur ZB-1', zb_a['ZB1_zuordnung'] and not zb_a['ZB2_lage'] and 'nicht aufloesbar' in zb_a['urteil'])
    def mit_diff(wert, lo, hi):
        return zielaussage_b({**Bblock, 'nks': {**nks, 'identifizierbar_anteil': {**nks['identifizierbar_anteil'], 'differenz': {'wert': wert, 'ci95': [lo, hi]}}}}, {'status': 'bestaetigend'}, CFG)
    z_a, z_b, z_c = mit_diff(0.04, 0.01, 0.07), mit_diff(0.20, 0.05, 0.35), mit_diff(0.12, -0.01, 0.20)
    z_d = mit_diff(0.20, 0.12, 0.30)
    check('Za = Ueberlegenheit (Untergrenze > 0); 10 Prozentpunkte nur beschreibend', z_a['Za_ueberlegenheit'] and not z_a['Za_differenz_ge_010_beschreibend'] and z_b['Za_ueberlegenheit'] and z_b['Za_differenz_ge_010_beschreibend']
          and not z_b['Za_untergrenze_ge_010_beschreibend'] and not z_c['Za_ueberlegenheit'] and 'nicht gestuetzt' in z_c['urteil'] and z_d['Za_untergrenze_ge_010_beschreibend'],
          f"{z_a['Za_ueberlegenheit']} {z_b['Za_untergrenze_ge_010_beschreibend']} {z_c['Za_ueberlegenheit']}")
    zb2 = zielaussage_b({**Bblock, 'pkb': {'e0': {'erfuellt': False, 'beurteilbar': True}}}, {'status': 'bestaetigend'}, CFG)
    zb3 = zielaussage_b(Bblock, {'status': 'pilot'}, CFG)
    zb5 = zielaussage_b({**Bblock, 'pkb': {'e0': {'erfuellt': False, 'beurteilbar': False}}}, {'status': 'bestaetigend'}, CFG)
    zb6 = zielaussage_b({**Bblock, 'nks': nks_ok}, {'status': 'bestaetigend', 'nachtraeglich': True}, CFG)
    check('ZB: E0 verfehlt/nicht beurteilbar -> nicht beurteilbar; Pilot -> kein Urteil; Freigabe nach H0 nie bestaetigend', 'nicht beurteilbar' in zb2['urteil'] and 'nicht beurteilbar' in zb5['urteil'] and 'kein Urteil' in zb3['urteil']
          and 'bestaetigt' not in zb6['urteil'] and 'nachtraeglich freigegeben' in zb6['urteil'], f"{zb6['urteil'][:80]}")
    # 18. H0: Populationsminimum, Aufschluesselung, Sperre
    Az = {'identifizierbar_anteil': {'wert': .9}, 'innerhalb_min': {'2': {'anteil': .97}}, 'population_minimum_bei_min': 0.0}
    check('H0 mit Populationsminimum', h0_pruefen(Az)['erfuellt'] and not h0_pruefen({**Az, 'population_minimum_bei_min': 10.0})['erfuellt']
          and not h0_pruefen({**Az, 'population_minimum_bei_min': None})['erfuellt'] and h0_pruefen({**Az, 'population_minimum_bei_min': -2.0})['erfuellt'])
    mix = _kunst_besuche(30, seed=21) + [kopie(b, id=b.id + 1000, mrn=b.mrn + 1000) for b in _kunst_besuche(30, seed=22, flach=True)]
    resM, DM = werte_aus(mix, 'A', 60)
    hv = h0_variabilitaet(resM, DM)
    check('H0-Aufschluesselung nach HF-Variabilitaet', hv['oben']['anteil'] > hv['unten']['anteil'] and hv['unten']['nicht_identifizierbar'] > 0 and 'terzilgrenzen_bpm' in hv,
          f"oben={hv['oben']['anteil']:.2f} unten={hv['unten']['anteil']:.2f}")
    # H0 oberes Terzil: alle drei Teile (Populationsminimum), Fehler 7
    Dg = raster(60, 2)
    def fake(sd, loss_fn):
        return {'nenner': True, 'ok': True, 'offset': 0.0, 'hf_sd': float(sd), '_loss': loss_fn(Dg), 'n': 5, 'id': sd, 'mrn': sd}
    fa_ok = [fake(i, lambda d: np.abs(d) + 0.0) for i in range(1, 31)]
    fa_pm = [fake(i, lambda d: np.abs(d - 10.0)) for i in range(1, 31)]
    check('H0-Terzilpruefung verlangt auch das Populationsminimum im oberen Terzil', h0_variabilitaet(fa_ok, Dg)['oben_erfuellt_h0'] and not h0_variabilitaet(fa_pm, Dg)['oben_erfuellt_h0']
          and h0_variabilitaet(fa_pm, Dg)['oben']['populationsminimum_bei_min'] == 10.0)
    gA = {'A': {'zusammenfassung': {**Az, 'breite_median_min': 8.0}, 'nk3': {'1': {'anteil': {'wert': .9}}, '5': {'anteil': {'wert': .8}}, '15': {'anteil': {'wert': .6}}, '30': {'anteil': {'wert': .5}},
                                                                              '60': {'anteil': {'wert': .3}}, '120': {'anteil': {'wert': .1}}},
                'nk1': {'identifizierbar_max': 3}, 'nk2': {'falsch_identifiziert_anteil_mittel': .02}, 'nk4': {'identifizierbar_anteil': {'wert': .01}}}}
    dd = deutung(gA, {'status': 'explorativ'}, 'bestaetigung')
    check('Deutung A: H0 H3 H4 H6 NK4', dd['H0']['erfuellt'] and dd['H3'] and dd['H4'] and dd['H6'] and dd['NK4'] and not dd['B_gesperrt_durch_H0'], str({k: v for k, v in dd.items() if k.startswith(('H', 'N'))}))
    ok5 = {k: {'beide_identifizierbar': 20, 'erfolg_s5_regel': True, 'median_abs_fehler_min': 1.0, 'absolut_median_abs_fehler_min': 1.0} for k in ('0', '10', '-10', '30', '-30')}
    h5a = deutung({'A': {'zusammenfassung': {}, 'pk2_verschiebung': ok5}}, {'status': 'explorativ'}, 'bestaetigung')
    ok4 = {k: v for k, v in ok5.items() if k != '30'}
    h5b = deutung({'A': {'zusammenfassung': {}, 'pk2_verschiebung': ok4}}, {'status': 'explorativ'}, 'bestaetigung')
    ok5n = dict(ok5, **{'30': {**ok5['30'], 'beide_identifizierbar': 0}})
    h5c = deutung({'A': {'zusammenfassung': {}, 'pk2_verschiebung': ok5n}}, {'status': 'explorativ'}, 'bestaetigung')
    ok5f = dict(ok5, **{'10': {**ok5['10'], 'median_abs_fehler_min': 9.0}})
    h5d = deutung({'A': {'zusammenfassung': {}, 'pk2_verschiebung': ok5f}}, {'status': 'explorativ'}, 'bestaetigung')
    check('H5 verlangt alle fuenf Bedingungen: fehlende/nicht auswertbare = nicht beurteilbar (None), nie still erfuellt', h5a['H5'] is True and h5b['H5'] is None and h5b['H5_nicht_beurteilbar_bedingungen'] == ['30']
          and h5c['H5'] is None and h5d['H5'] is False)
    gA['A']['zusammenfassung']['population_minimum_bei_min'] = 12.0
    dd2 = deutung({**gA, 'B': {'gesperrt': 'H0'}}, {'status': 'explorativ'}, 'bestaetigung')
    check('H0-Sperre: B ohne Urteil bei verfehltem Populationsminimum', dd2['B_gesperrt_durch_H0'] and 'gesperrt' in dd2['ZB']['urteil'])
    gA['A']['nk3']['60']['anteil']['wert'] = .5
    check('H3 verfehlt bei 60-min-Wert ueber 40 %', not deutung(gA, {'status': 'explorativ'}, 'bestaetigung')['H3'])
    # 19. Sensitivitaeten im Hauptlauf (A und B)
    sens = sensitivitaeten(bB)
    check('Sensitivitaeten A (G30, G120, 900 s, max6, E0)', set(sens['A']) == {'G30_schritt1', 'G120_schritt2', 'abstand900', 'max6', 'E0'} and sens['A']['G30_schritt1']['nenner'] > 0, str({k: v['nenner'] for k, v in sens['A'].items()}))
    check('Sensitivitaeten B (Referenz 60 min, 900 s, max6, E0, ohne Anker, drei Anker)', set(sens['B']) == {'abstand900', 'max6', 'E0', 'referenzabstand_60min', 'ohne_ankerbedingung', 'drei_anker'}
          and sens['B']['ohne_ankerbedingung']['nenner'] >= SB['nenner'] >= sens['B']['drei_anker']['nenner'], str({k: v['nenner'] for k, v in sens['B'].items()}))
    bv = besuch_variante(kopie(bB[0], seg=np.repeat([0, 1], len(bB[0].ts) // 2 + 1)[:len(bB[0].ts)]), 'abstand900', 'B')
    check('abstand900 nimmt jede dritte Einheit je Segment', len(bv.ts) == int(np.ceil(np.sum(np.repeat([0, 1], len(bB[0].ts) // 2 + 1)[:len(bB[0].ts)] == 0) / 3)) + int(np.ceil(np.sum(np.repeat([0, 1], len(bB[0].ts) // 2 + 1)[:len(bB[0].ts)] == 1) / 3)))
    check('max6 nimmt hoechstens 6 Einheiten', len(besuch_variante(bB[0], 'max6', 'B').ts) == 6)
    # 20. Erkenner-Kriterium
    bE = [kopie(b, hfA=b.hfA.copy()) for b in _kunst_besuche(10, seed=31)]
    for b in bE:
        b.mon_t = np.arange(b.lo, b.hi, 60.0)
        b.mon_v = np.full(len(b.mon_t), 80.0)
        b.hfA = np.full(len(b.ts), 82.0)
    k1 = erkenner_kriterium(bE)
    for b in bE:
        b.hfA[::2] = 95.0
    k2 = erkenner_kriterium(bE)
    check('Erkenner-Kriterium (5/min, 95 %)', k1['kriterium_erfuellt'] and not k2['kriterium_erfuellt'] and k2['anteil_innerhalb'] < 0.6, f"{k1['anteil_innerhalb']:.2f} {k2['anteil_innerhalb']:.2f}")
    bE2 = [kopie(b, hfA=np.where(np.arange(len(b.ts)) % 10 < 4, np.nan, 82.0)) for b in bE]
    k3 = erkenner_kriterium(bE2)
    check('Erkenner: hohe Uebereinstimmung, aber Ausbeute unter der Mindestausbeute -> Kriterium verfehlt', k3['anteil_erfuellt'] and not k3['ausbeute_erfuellt'] and not k3['kriterium_erfuellt'] and k3['ausbeute'] < 0.8,
          f"anteil={k3['anteil_innerhalb']:.2f} ausbeute={k3['ausbeute']:.2f}")
    kr = lambda an, au: {'anteil_innerhalb': an, 'ausbeute': au, 'kriterium_erfuellt': an >= 0.95 and au >= 0.8}
    vv = {'vollstaendig': True, 'grundgesamtheit': {'sha256': 'H1'}}
    eint = [{'ev': 'A', 'utc': '1', **vv, 'kriterium': kr(0.97, 0.9)}, {'ev': 'B', 'utc': '2', **vv, 'kriterium': kr(0.99, 0.7)}, {'ev': 'C', 'utc': '3', **vv, 'kriterium': kr(0.98, 0.85)},
            {'ev': 'D', 'utc': '4', **vv, 'kriterium': kr(0.98, 0.9)}, {'ev': 'E', 'utc': '0', **vv, 'kriterium': kr(0.98, 0.9)}]
    check('Erkennerwahl nur nach technischer Guete: Mindestausbeute, hoechste Uebereinstimmung, dann Ausbeute, dann frueheste', waehle_erkenner(eint) == 'E' and waehle_erkenner(eint[:3]) == 'C'
          and waehle_erkenner([eint[1]]) is None and waehle_erkenner([]) is None)

    # 20b. Festhash (ast): jeder Modulcode ausser den zwei Erkennerfunktionen und dem Konfigurationsblock ist gebunden; Helfer in der Erkennerversion
    src = Path(__file__).read_text(encoding='utf-8')
    with tempfile.TemporaryDirectory() as td2:
        td2 = Path(td2)

        def h_von(text, name):
            p_ = td2 / name
            p_.write_text(text, encoding='utf-8')
            return fest_hash(p_)
        h0_ = h_von(src, 'a0.py')
        erlaubt = src.replace("    x = np.nan_to_num(np.asarray(x, float), nan=0.0)\n", "    x = np.nan_to_num(np.asarray(x, float), nan=0.0)  # Erkenner geaendert\n    x = x * 1.0\n", 1).replace("'rr_cv_max': 0.5,", "'rr_cv_max': 0.4,", 1)
        check('Festhash: Aenderung von Erkennerfunktion und Konfiguration aendert den Festhash nicht', erlaubt != src and h_von(erlaubt, 'a1.py') == h0_ == fest_hash())
        check('Festhash ist strukturell: Kommentare und Leerzeilen aendern ihn nicht', h_von(src.replace("\n\n\ndef schaetze(", "\n\n\n# Kommentar\n\n\n\ndef schaetze(", 1), 'a2.py') == h0_)
        verboten = {'deutung': src.replace("out['H6'] = bool(", "out['H6'] = not bool(", 1),
                    'erkenner_kriterium': src.replace("n_ok += abs(h - ref) <= cfg['erkenner']['innerhalb_bpm']", "n_ok += abs(h - ref) <= 99", 1),
                    'modus_auswerten': src.replace("    h0_ok = h0_pruefen(A)['erfuellt']\n    status_urteil = status", "    h0_ok = True\n    status_urteil = status", 1),
                    'waehle_erkenner_eintrag': src.replace("e.get('utc', '')))\n    return ok[0]", "e.get('utc', ''), 1))\n    return ok[0]", 1),
                    'kontrolle_permutation': src.replace("rng = np.random.default_rng(cfg['seeds']['nk1_permutation'])", "rng = np.random.default_rng(1)", 1),
                    'modus_einfrieren': src.replace("if best['ev'] != ev or", "if False and best['ev'] != ev or", 1),
                    # Modulcode unmittelbar hinter der Erkennerfunktion (Sol F2): erweitert die erlaubten Aenderungen, darf nicht ausgeschlossen sein
                    'code_hinter_hf_fenster': src.replace("    return float(hf), n\n", "    return float(hf), n\nPILOT_ERLAUBT['hf_max'] = (0.0, 999.0)\n", 1),
                    'pilot_erlaubt_erweitert': src.replace("'abdeckung_A_min_s': (45.0, 54.0)", "'abdeckung_A_min_s': (0.0, 99.0)", 1),
                    'extr_keys': src.replace("EXTR_KEYS = ['excl_s',", "EXTR_KEYS = ['hf_max', 'excl_s',", 1),
                    'import_zeile': src.replace("import numpy as np\n", "import numpy as np\nimport random\n", 1)}
        diffs = {k: (v != src and h_von(v, f'v_{k}.py') != h0_) for k, v in verboten.items()}
        check('Festhash: Aenderungen an Deutung, Kriterium, Ablauf, Auswahlregel, Kontrollen, Einfrieren UND an Modulcode hinter/neben den Erkennerfunktionen werden erkannt', all(diffs.values()), str(diffs))
        try:
            h_von(src.replace("def r_zacken(", "def r_zacken_x(", 1), 'fehlt.py')
            fehlt_ok = False
        except SystemExit:
            fehlt_ok = True
        check('Festhash: fehlende Erkennerfunktion ist ein Fehler (keine stille Abgrenzung)', fehlt_ok)
        hz = td2 / 'helfer.py'
        hz.write_text(HELFER_PATH.read_text(encoding='utf-8') + '\n# geaendert\n', encoding='utf-8')
        alt = globals()['HELFER_PATH']
        globals()['HELFER_PATH'] = hz
        try:
            ev_h, hh_ = erkenner_version(), helfer_hash()
        finally:
            globals()['HELFER_PATH'] = alt
        check('Helfer s5_mcmed_zaehlung.py: geaenderter Helfer = neue Erkennerversion und anderer Helferhash', ev_h != erkenner_version() and hh_ != helfer_hash())

    # 21. Sperre, Bindung, Einfrieren, Hashes (Temp-Ordner, Kopie der Spezifikation)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        spec = td / 'spec.md'
        spec.write_text('Test-Spezifikation', encoding='utf-8')
        lk, fr = td / 'lock.json', td / 'freeze.json'
        try:
            pruefe_start('pilot', lk, fr, spec)
            fehl = False
        except SystemExit:
            fehl = True
        check('Start ohne Bindung gesperrt', fehl)
        modus_sperren(spec, lk)
        check('Start nach Bindung (Pilot) erlaubt', pruefe_start('pilot', lk, fr, spec)['quelle'] == 'Bindung')

        def gesperrt(*a, **k):
            try:
                pruefe_start(*a, **k)
                return False
            except SystemExit:
                return True
        spec2 = td / 'spec2.md'
        spec2.write_text('veraendert', encoding='utf-8')
        check('Spezifikationshash geprueft', gesperrt('pilot', lk, fr, spec2))
        check('Bestaetigung ohne Einfrieren gesperrt', gesperrt('bestaetigung', lk, fr, spec))
        cfg_x = copy.deepcopy(CFG)
        cfg_x['B']['G_min'] = 30
        check('Konfigurationshash geprueft (G fuer B geaendert)', gesperrt('pilot', lk, fr, spec, cfg_x))
        lkd = json.loads(lk.read_text(encoding='utf-8'))
        lkd['code_sha256'] = '0' * 64
        lk2 = td / 'lock2.json'
        lk2.write_text(json.dumps(lkd), encoding='utf-8')
        check('Codehash geprueft', gesperrt('pilot', lk2, fr, spec))
        lkd['code_sha256'] = code_hash()
        lkd['fest_sha256'] = '0' * 64
        lk3 = td / 'lock3.json'
        lk3.write_text(json.dumps(lkd), encoding='utf-8')
        check('Kernhash geprueft', gesperrt('pilot', lk3, fr, spec))
        cfg_bad = copy.deepcopy(CFG)
        cfg_bad['hf_max'] = 300.0
        try:
            modus_einfrieren('t', lk, fr, spec, cfg_bad)
            verw = False
        except SystemExit:
            verw = True
        cfg_bad2 = copy.deepcopy(CFG)
        cfg_bad2['rr_cv_max'] = 0.6
        try:
            modus_einfrieren('t', lk, fr, spec, cfg_bad2)
            verw2 = False
        except SystemExit:
            verw2 = True
        check('Einfrieren verweigert (Parameter ausserhalb / Spanne verlassen)', verw and verw2 and not fr.exists())
        lkn = json.loads(lk.read_text(encoding='utf-8'))
        lkn['numpy_version'] = '0.0.0'
        lk4 = td / 'lock4.json'
        lk4.write_text(json.dumps(lkn), encoding='utf-8')
        check('numpy-Version der Bindung geprueft (Zufallsstrom)', gesperrt('pilot', lk4, fr, spec))
        cfg_ok = copy.deepcopy(CFG)
        cfg_ok['rr_cv_max'] = 0.4
        ev_ok = erkenner_version(cfg_ok)
        pr = td / 'prot.jsonl'
        import duckdb as _dd

        def mini_db(erledigt):
            cx = _dd.connect()
            cx.execute("create table visit(CSN bigint,MRN bigint,split varchar,r double,d double)")
            cx.execute("create table hd2(CSN bigint,seg int,fs double,n bigint,t0 double,t1 double)")
            cx.execute("insert into visit values (1,1,'val',0,100),(2,2,'val',0,100),(3,3,'val',0,100),(4,4,'val',0,100),(5,5,'train',0,100)")
            cx.execute("insert into hd2 values (1,1,500,10,0,1),(2,1,500,10,0,1),(3,1,500,10,0,1),(4,1,500,10,0,1),(5,1,500,10,0,1)")
            tabellen_anlegen(cx)
            for csn_ in erledigt:
                cx.execute("insert into extr_done_v16 values (?,?)", [csn_, ev_ok])
            return cx
        cdb = mini_db([1, 2, 3, 4])
        gg_ = val_grundgesamtheit(cdb)
        vg = {'vollstaendig': True, 'grundgesamtheit': gg_}
        check('Val-Grundgesamtheit: Zahl und Hash der Besuchsmenge (ohne Train)', gg_['besuche'] == 4 and len(gg_['sha256']) == 64 and val_grundgesamtheit(mini_db([]))['sha256'] == gg_['sha256'])

        def einf_verweigert(prot):
            try:
                modus_einfrieren('Test', lk, fr, spec, cfg_ok, prot, c=cdb)
                return False
            except SystemExit:
                return True
        check('Einfrieren ohne Erkenner-Protokoll verweigert', einf_verweigert(pr) and not fr.exists())
        h_ok = {'code_sha256': code_hash(), 'cfg_sha256': cfg_hash(cfg_ok)}
        pr.write_text(json.dumps({'ev': ev_ok, 'utc': '2', **h_ok, **vg, 'kriterium': {'anteil_innerhalb': 0.97, 'ausbeute': 0.7, 'kriterium_erfuellt': False}}) + '\n', encoding='utf-8')
        check('Einfrieren verweigert, wenn die Mindestausbeute verfehlt ist', einf_verweigert(pr) and not fr.exists())
        with open(pr, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'ev': 'andere', 'utc': '1', **vg, 'kriterium': {'anteil_innerhalb': 0.99, 'ausbeute': 0.95, 'kriterium_erfuellt': True}}) + '\n')
            f.write(json.dumps({'ev': ev_ok, 'utc': '3', **h_ok, **vg, 'kriterium': {'anteil_innerhalb': 0.97, 'ausbeute': 0.9, 'kriterium_erfuellt': True}}) + '\n')
        check('Einfrieren verweigert, wenn eine andere Version technisch besser ist (Auswahlregel)', einf_verweigert(pr) and not fr.exists())
        pr2 = td / 'prot2.jsonl'
        pr3 = td / 'prot3.jsonl'
        pr3.write_text(json.dumps({'ev': ev_ok, 'utc': '3', 'code_sha256': '0' * 64, 'cfg_sha256': h_ok['cfg_sha256'], **vg, 'kriterium': {'anteil_innerhalb': 0.97, 'ausbeute': 0.9, 'kriterium_erfuellt': True}}) + '\n', encoding='utf-8')
        check('Einfrieren verweigert, wenn der protokollierte Kandidat einen anderen Code-/Konfigurationshash traegt', einf_verweigert(pr3) and not fr.exists())
        # Kandidatenmodus: erlaubte Aenderung darf vor dem Einfrieren starten, unerlaubte nicht, nach dem Einfrieren keine Kandidaten mehr
        check('Kandidatenstufe: erlaubte Erkenneraenderung (rr_cv_max 0,4) erlaubt, Festhash/Helfer/unerlaubte Parameter gesperrt', pruefe_start('kandidat', lk, fr, spec, cfg_ok)['kandidat']
              and gesperrt('kandidat', lk, fr, spec, cfg_bad) and gesperrt('kandidat', lk, fr, spec, cfg_bad2) and gesperrt('pilot', lk, fr, spec, cfg_ok) and gesperrt('kandidat', lk3, fr, spec))
        lkh = json.loads(lk.read_text(encoding='utf-8'))
        lkh['helfer_sha256'] = '0' * 64
        lk5 = td / 'lock5.json'
        lk5.write_text(json.dumps(lkh), encoding='utf-8')
        check('Helfer s5_mcmed_zaehlung.py in der Versionsbindung (Abweichung = gesperrt)', gesperrt('pilot', lk5, fr, spec) and gesperrt('kandidat', lk5, fr, spec))
        lkb = json.loads(lk.read_text(encoding='utf-8'))
        lkb['basis_spec_sha256'] = '0' * 64
        lk6 = td / 'lock6.json'
        lk6.write_text(json.dumps(lkb), encoding='utf-8')
        check('Basis-Spezifikation (v1.2) in der Bindung geprueft', gesperrt('pilot', lk6, fr, spec))
        # Basis zwingend: fehlende Datei oder fehlender Hash wird zurueckgewiesen (Bindung und Start)
        nichtda = td / 'nicht_vorhanden.md'
        try:
            modus_sperren(spec, td / 'lockX.json', None, nichtda)
            sp_ok = False
        except SystemExit:
            sp_ok = True
        check('Bindung ohne Basisdatei verweigert (v1.2); keine Sperrdatei geschrieben', sp_ok and not (td / 'lockX.json').exists())
        try:
            modus_sperren(spec, td / 'lockY.json', None, None, nichtda)
            sp3_ok = False
        except SystemExit:
            sp3_ok = True
        check('Bindung ohne Basisdatei verweigert (v1.3)', sp3_ok and not (td / 'lockY.json').exists())
        zurueck = {}
        for nm, mod in (('ohne12', lambda d: d.pop('basis_spec_sha256')), ('null12', lambda d: d.update(basis_spec_sha256=None)), ('leer12', lambda d: d.update(basis_spec_sha256='')),
                        ('ohne13', lambda d: d.pop('basis13_spec_sha256')), ('falsch13', lambda d: d.update(basis13_spec_sha256='0' * 64)),
                        ('ohne14', lambda d: d.pop('basis14_spec_sha256')), ('falsch14', lambda d: d.update(basis14_spec_sha256='0' * 64)),
                        ('ohne15', lambda d: d.pop('basis15_spec_sha256')), ('falsch15', lambda d: d.update(basis15_spec_sha256='0' * 64))):
            dd_ = json.loads(lk.read_text(encoding='utf-8'))
            mod(dd_)
            pth = td / f'lock_{nm}.json'
            pth.write_text(json.dumps(dd_), encoding='utf-8')
            zurueck[nm] = gesperrt('pilot', pth, fr, spec) and gesperrt('kandidat', pth, fr, spec) and gesperrt('bestaetigung', pth, fr, spec)
        check('Start ohne Basishash in der Bindung (fehlend, null, leer) oder mit falschem Hash zurueckgewiesen', all(zurueck.values()), str(zurueck))
        check('Start mit fehlender Basisdatei zurueckgewiesen', gesperrt('pilot', lk, fr, spec, None, nichtda) and gesperrt('pilot', lk, fr, spec, None, None, nichtda))
        # Erkennerwahl nur auf der vollstaendigen Val-Grundgesamtheit
        gg_fremd = {'besuche': 4, 'sha256': '1' * 64}
        pr2.write_text(json.dumps({'ev': ev_ok, 'utc': '3', **h_ok, **vg, 'kriterium': {'anteil_innerhalb': 0.97, 'ausbeute': 0.9, 'kriterium_erfuellt': True}}) + '\n', encoding='utf-8')
        with open(pr2, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'ev': 'teilbestand', 'utc': '1', 'vollstaendig': False, 'grundgesamtheit': {'besuche': 2, 'sha256': '2' * 64}, 'kriterium': {'anteil_innerhalb': 0.999, 'ausbeute': 0.99, 'kriterium_erfuellt': True}}) + '\n')
            f.write(json.dumps({'ev': 'ohne_vollstaendig_feld', 'utc': '1', 'grundgesamtheit': gg_, 'kriterium': {'anteil_innerhalb': 0.999, 'ausbeute': 0.99, 'kriterium_erfuellt': True}}) + '\n')
            f.write(json.dumps({'ev': 'andere_grundgesamtheit', 'utc': '1', 'vollstaendig': True, 'grundgesamtheit': gg_fremd, 'kriterium': {'anteil_innerhalb': 0.999, 'ausbeute': 0.99, 'kriterium_erfuellt': True}}) + '\n')
        eint2 = lies_protokoll(pr2)
        check('Erkennerwahl ignoriert Teilbestands-, unvollstaendig markierte und fremde Grundgesamtheits-Eintraege', waehle_erkenner(eint2, gg_['sha256']) == ev_ok and waehle_erkenner(eint2) in (ev_ok, 'andere_grundgesamtheit')
              and waehle_erkenner_eintrag([e for e in eint2 if e['ev'] in ('teilbestand', 'ohne_vollstaendig_feld')], gg_['sha256']) is None)
        cdb_teil = mini_db([1, 2, 3])
        try:
            modus_einfrieren('Test', lk, fr, spec, cfg_ok, pr2, c=cdb_teil)
            teil_ok = False
        except SystemExit:
            teil_ok = True
        check('Einfrieren verweigert, wenn die aktuelle Erkennerversion nicht auf der vollstaendigen Val-Grundgesamtheit extrahiert ist', teil_ok and not fr.exists())
        modus_einfrieren('Test', lk, fr, spec, cfg_ok, pr2, c=cdb)
        check('Nach dem Einfrieren keine Erkennerkandidaten mehr', gesperrt('kandidat', lk, fr, spec, cfg_ok))
        check('Einfrieren erlaubt (rr_cv_max 0,4); danach gilt das Einfrieren', fr.exists() and gesperrt('pilot', lk, fr, spec) and pruefe_start('pilot', lk, fr, spec, cfg_ok)['quelle'] == 'Einfrieren')
        check('Bestaetigung nach Einfrieren erlaubt', pruefe_start('bestaetigung', lk, fr, spec, cfg_ok)['quelle'] == 'Einfrieren')
    # 22. extr_done an Erkennerversion gebunden
    import duckdb
    ev1 = erkenner_version()
    cfg_v = copy.deepcopy(CFG)
    cfg_v['rr_cv_max'] = 0.4
    ev2 = erkenner_version(cfg_v)
    c = duckdb.connect()
    c.execute("create table visit(CSN bigint,MRN bigint,split varchar,r double,d double)")
    c.execute("create table hd2(CSN bigint,seg int,fs double,n bigint,t0 double,t1 double)")
    c.execute("insert into visit values (1,1,'val',0,100),(2,2,'val',0,100),(3,3,'train',0,100)")
    c.execute("insert into hd2 values (1,1,500,10,0,1),(2,1,500,10,0,1),(3,1,500,10,0,1)")
    tabellen_anlegen(c)
    c.execute("insert into extr_done_v16 values (1,?),(2,?)", [ev1, ev1])
    check('Erkennerversion aendert sich mit Parametern/Erkenner', ev1 != ev2 and len(ev1) == 16)
    check('extr_done: gleiche Version erledigt, andere Version wieder offen, Split getrennt',
          offene_besuche(c, 'val', ev1) == [] and offene_besuche(c, 'val', ev2) == [1, 2] and offene_besuche(c, 'train', ev1) == [3])
    # 23. Ausgabe maskiert Kleinzellen (Struktur wie im Hauptlauf)
    m = Z.mask_tree({'status': zweig_bestimmen(7, 4, 'bestaetigung'), 'x': {'nenner': 5}})
    dz3 = deutung({'A': {'zusammenfassung': {}}, 'B': {'nicht_auswertbar': True, 'beschreibung_ohne_urteil': {}}}, {'status': 'nicht_auswertbar'}, 'bestaetigung')
    check('Zweig Z3: kein ZB-Urteil, nur Beschreibung', 'kein ZB-Urteil' in dz3['ZB']['urteil'] and 'Beschreibung' in dz3['ZB']['urteil'])
    check('Kleinzellenmaske in Status', m['status']['rotierbarer_nenner'] == '<10' and m['status']['rotierbare_patienten'] == '<10' and m['x']['nenner'] == '<10')
    print('SELBSTTEST', 'BESTANDEN' if ok else 'FEHLGESCHLAGEN', f'({n_pruef} Pruefungen)', 'code_sha256=' + code_hash(), 'fest_sha256=' + fest_hash()[:16])
    return ok


if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else ''
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    if m == 'selbsttest':
        sys.exit(0 if selbsttest() else 1)
    elif m == 'sperren':
        modus_sperren()
    elif m == 'einfrieren':
        modus_einfrieren(arg('--grund', 'ohne Angabe'))
    elif m == 'erkenner':
        modus_erkenner()
    elif m == 'extrahieren':
        modus_extrahieren(arg('--split'), int(arg('--max-besuche')) if arg('--max-besuche') else None)
    elif m == 'pilot':
        modus_auswerten('val', 'pilot', b_trotz_h0='--b-trotz-h0' in sys.argv)
    elif m == 'bestaetigung':
        modus_auswerten('train', 'bestaetigung', b_trotz_h0='--b-trotz-h0' in sys.argv)
    else:
        print(__doc__)
        sys.exit(2)
