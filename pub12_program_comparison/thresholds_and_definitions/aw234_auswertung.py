# -*- coding: utf-8 -*-
"""aw234_auswertung.py (AW-234, Teil 1, Version 1.0): zusaetzliche explorative Analysen fuer PUB-12 auf PTB-XL P0 (offene Daten).

Population P0 (je Person die niedrigste ecg_id, 18.869 Personen), native Ausgaben, gemeinsamer Nenner je Paar (beide Werkzeuge mit gueltigem Wert).
Werkzeuge: LoopLens, BRAVEHEART, 12SL, Uni-G, ECGDeli (P0 voll); NeuroKit2 nur auf S_G (1.000 Personen, Designgewichte, g1 bis g3), nur QRS.
Rechenbasis (Laden, Hashpruefung der gesperrten Tabellen, Personencluster-Ziehungen) ist schwellen_aw228.py Version 1.3, unveraendert importiert
(SHA-256 wird vor dem Import geprueft). Gesperrte Tabellen werden nur gelesen, vorher stimmen alle Hashes (Datensperre 15.10.2026).

Analysen (alle explorativ, beschreibend, keine Hypothesentests):
  (a) Kreuztabelle der QRS-Klassen <120, 120 bis 129, 130 bis 149, >=150 ms je Paar (Zellanteile in Prozent des gemeinsamen Nenners) und Anteil der
      Klassenwechsel (Klasse Kandidat ungleich Klasse Komparator; getrennt nach Kandidat hoeher / niedriger).
  (b) dieselbe Diskordanz in der CLBBB-Teilmenge (SCP-Code CLBBB; Definition wie in bench_auswertung_v11.py / V2 4.2: Schluessel im Feld scp_codes
      vorhanden, ohne Wahrscheinlichkeitsschwelle; Empfindlichkeitsfassung Wahrscheinlichkeit >= 50) und in deren Komplement.
  (c) je Paar Bland-Altman (Bias, Standardabweichung der Differenz, Grenzen 1,96 SD), ICC(2,1) (zweiseitig zufaellig, absolute Uebereinstimmung,
      Einzelmessung, aus den Quadratsummen der Zwei-Wege-Varianzanalyse) und Cohens Kappa an 120 ms (mit beobachteter Uebereinstimmung und PABAK).
  (d) QTc mit den gelieferten Standardformeln je Werkzeug (12SL QT_IntCorr_Global = Bazett, Uni-G QT_IntCorr_Global = Hodge, ECGDeli
      QT_IntFramingham_Global = Framingham, LoopLens und BRAVEHEART qtcf = Fridericia nativ) und zusaetzlich einheitlich Fridericia (aus dem globalen QT
      und der eigenen RR-Quelle des Werkzeugs neu berechnet). Schwellen geschlechtsspezifisch (Maenner 450/470/500, Frauen 460/480/500; sex 0 = maennlich)
      und geschlechtsneutral (450/470/500).
  (e) Diskordanz nach Geschlecht (sex 0 = maennlich, 1 = weiblich), Differenz weiblich minus maennlich mit Intervall aus denselben Ziehungen.
Statistik: Diskordanz = Anteil der Personen im gemeinsamen Nenner, bei denen genau ein Werkzeug die Schwelle erreicht (Wert >= Schwelle).
Differenz = Kandidat minus Komparator (LoopLens ist Kandidat, sonst das in ORDNUNG spaetere Werkzeug). Intervalle: Perzentil-Bootstrap (2,5 und 97,5 %),
Personencluster, 2.000 Ziehungen, fester Seed, gemeinsame Ziehungen fuer alle Einheiten; zufaelliger Nenner je Ziehung. Alle Kennzahlen sind
Funktionen gewichteter Mittelwerte und werden je Ziehung aus diesen berechnet (bei S_G mit Designgewichten N_h / n_h; Varianz-, ICC- und Kappa-Formeln dort
naeherungsweise mit der Gewichtssumme als Stichprobenumfang). Keine Maskierung (offene Daten, AE-296); Methods nennt das.

Aufruf (Python der Auswertung, ohne Bytecode):
  <local path> -I -B aw234_auswertung.py --modus hash|lauf
Test (Kunstdaten):  ... -I -B aw234_auswertung_test.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

VERSION = "1.0"
SEED = 15102340
SEED_SG = 15102341
B_ZIEHUNGEN = 2000
CHUNK = 250
S_PFAD = Path(str(data_path("pub12.werkzeuge_pub12_schwellen_aw228_py")))
S_HASH = "889b846c59073a009a635da2883f9795ee922fda98ea719be09ec6106ffb864d"  # schwellen_aw228.py Version 1.3 (schwellen_aw228_eingefroren_v1.3.txt)
OUT = Path(str(data_path("pub12.benchmark_pub12_aw234_auswertung")))
KOPIE = Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse_aw234")))
WERKZEUGE = ["LoopLens", "BRAVEHEART", "12SL", "Uni-G", "ECGDeli"]
KLASSEN = ("<120", "120-129", "130-149", ">=150")
QRS_SCHWELLEN = (120, 130, 150)
QTC_SS = {"ss450_460": (450, 460), "ss470_480": (470, 480), "ss500_500": (500, 500)}  # Maenner, Frauen
QTC_NEUTRAL = {"n450": 450, "n470": 470, "n500": 500}
QTC_QUELLE = ("Rautaharju PM, Surawicz B, Gettes LS, et al. AHA/ACCF/HRS recommendations for the standardization and interpretation of the "
              "electrocardiogram: part IV: the ST segment, T and U waves, and the QT interval. Circulation 2009;119:e241-e250. PMID 19228821 "
              "(parallel veroeffentlicht in J Am Coll Cardiol 2009, PMID 19281931). PMID gegen PubMed geprueft; die Zahlenstufen 450/470/500 "
              "(Maenner) und 460/480/500 (Frauen) sind die Vorgabe des Auftrags AW-234, im Volltext der Quelle nicht nachgelesen.")


def _lade_modul():
    h = hashlib.sha256(S_PFAD.read_bytes()).hexdigest()
    if h != S_HASH:
        raise SystemExit("schwellen_aw228.py weicht von Version 1.3 ab: %s" % h)
    spec = importlib.util.spec_from_file_location("schwellen_aw228", str(S_PFAD))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["schwellen_aw228"] = mod
    spec.loader.exec_module(mod)
    return mod


S = _lade_modul()
log = S.log
ki95 = S.ki95


# ---------------------------------------------------------------------------------------------------------------
# 1. Kennzahlen aus gewichteten Mittelwerten
# ---------------------------------------------------------------------------------------------------------------
def klasse(x):
    """QRS-Klasse 0..3 (<120, 120-129, 130-149, >=150)."""
    x = np.asarray(x, float)
    return (x >= 120).astype(int) + (x >= 130).astype(int) + (x >= 150).astype(int)


class Einheit:
    """Ein Paar (Kandidat a, Komparator b) auf einer Teilmenge. a, b, w, maske haben die Laenge der Personenmenge (K).
    schwellen: dict Name -> Zahl oder Array (Laenge K, personenbezogene Schwelle). klassen: QRS-Kreuztabelle (16 Zellen)."""

    def __init__(self, name, a, b, w, maske, schwellen, klassen=False, meta=None):
        self.name = name
        self.meta = meta or {}
        a, b, w = np.asarray(a, float), np.asarray(b, float), np.asarray(w, float)
        valid = np.isfinite(a) & np.isfinite(b) & np.asarray(maske, bool)
        self.idx = np.flatnonzero(valid)
        a, b, self.w = a[self.idx], b[self.idx], w[self.idx]
        self.a, self.b = a, b
        d = a - b
        m = (a + b) / 2.0
        cols, namen = [d, d * d, m, m * m], ["d", "d2", "m", "m2"]
        self.schwellen = list(schwellen)
        for lab in self.schwellen:
            t = schwellen[lab]
            t = np.asarray(t, float)[self.idx] if np.ndim(t) else float(t)
            pa, pb = a >= t, b >= t
            for k, x in (("pa", pa), ("pb", pb), ("disc", pa ^ pb), ("nur_a", pa & ~pb), ("nur_b", ~pa & pb)):
                cols.append(x.astype(float))
                namen.append("%s|%s" % (k, lab))
        self.klassen = bool(klassen)
        if klassen:
            ca, cb = klasse(a), klasse(b)
            for i in range(4):
                for j in range(4):
                    cols.append(((ca == i) & (cb == j)).astype(float))
                    namen.append("z|%d|%d" % (i, j))
        self.namen = namen
        self.ix = {n: i for i, n in enumerate(namen)}
        self.X = np.column_stack(cols).astype(float)
        self.n = len(self.idx)

    def ableiten(self, M, N):
        """Kennzahlen je Zeile von M (gewichtete Mittelwerte der Spalten) und Gewichtssumme N."""
        ix = self.ix
        out = {"n_gewicht": N}
        with np.errstate(invalid="ignore", divide="ignore"):
            dbar = M[:, ix["d"]]
            v = (M[:, ix["d2"]] - dbar ** 2) * N / (N - 1.0)
            sd = np.sqrt(np.maximum(v, 0.0))
            out["bias"], out["sd_diff"] = dbar, sd
            out["loa_u"], out["loa_o"] = dbar - 1.96 * sd, dbar + 1.96 * sd
            g = M[:, ix["m"]]
            msr = 2.0 * N * (M[:, ix["m2"]] - g ** 2) / (N - 1.0)
            mse = N * (M[:, ix["d2"]] - dbar ** 2) / 2.0 / (N - 1.0)
            msc = N * dbar ** 2 / 2.0
            out["icc21"] = (msr - mse) / (msr + mse + 2.0 * (msc - mse) / N)
            for lab in self.schwellen:
                pa, pb, disc = M[:, ix["pa|" + lab]], M[:, ix["pb|" + lab]], M[:, ix["disc|" + lab]]
                po = 1.0 - disc
                pe = pa * pb + (1.0 - pa) * (1.0 - pb)
                out["prev_a|" + lab], out["prev_b|" + lab], out["disc|" + lab] = pa, pb, disc
                out["nur_a|" + lab], out["nur_b|" + lab] = M[:, ix["nur_a|" + lab]], M[:, ix["nur_b|" + lab]]
                out["po|" + lab] = po
                out["kappa|" + lab] = np.where(pe < 1.0, (po - pe) / (1.0 - pe), np.nan)
                out["pabak|" + lab] = 2.0 * po - 1.0
            if self.klassen:
                hi = lo = 0.0
                diag = 0.0
                for i in range(4):
                    for j in range(4):
                        z = M[:, ix["z|%d|%d" % (i, j)]]
                        out["z|%d|%d" % (i, j)] = z
                        diag = diag + z if i == j else diag
                        hi = hi + z if i > j else hi
                        lo = lo + z if i < j else lo
                out["klassenwechsel"] = 1.0 - diag
                out["wechsel_kandidat_hoeher"], out["wechsel_kandidat_niedriger"] = hi, lo
        return out

    def punkt(self):
        W = self.w[None, :]
        N = W.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            M = (W @ self.X) / N[:, None]
        return {k: float(v[0]) for k, v in self.ableiten(M, N).items()}

    def zaehlungen(self):
        z = {"n": int(self.n)}
        for lab in self.schwellen:
            for k in ("pa", "pb", "disc", "nur_a", "nur_b"):
                z["%s|%s" % (k, lab)] = int(self.X[:, self.ix["%s|%s" % (k, lab)]].sum())
        if self.klassen:
            for i in range(4):
                for j in range(4):
                    z["z|%d|%d" % (i, j)] = int(self.X[:, self.ix["z|%d|%d" % (i, j)]].sum())
        return z


def boot_lauf(einheiten, zieh):
    res = {e.name: {} for e in einheiten}
    for c in range(zieh.n_chunks):
        C = zieh.chunk_c(c)
        for e in einheiten:
            W = C[:, e.idx] * e.w[None, :]
            N = W.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                M = (W @ e.X) / N[:, None]
            for k, v in e.ableiten(M, N).items():
                res[e.name].setdefault(k, []).append(v)
    return {n: {k: np.concatenate(v) for k, v in d.items()} for n, d in res.items()}


def icc21_direkt(a, b):
    """ICC(2,1) direkt aus der Zwei-Wege-Varianzanalyse einer n x 2 Matrix (Kontrolle fuer die Momentenformel)."""
    X = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    n, k = X.shape
    g = X.mean()
    ssr = k * ((X.mean(axis=1) - g) ** 2).sum()
    ssc = n * ((X.mean(axis=0) - g) ** 2).sum()
    sst = ((X - g) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


# ---------------------------------------------------------------------------------------------------------------
# 2. Schwellen und Einheiten
# ---------------------------------------------------------------------------------------------------------------
def schwellen_qtc(sex):
    """Schwellenwoerterbuecher fuer QTc: geschlechtsspezifisch (personenbezogene Arrays) und neutral. sex 0 = maennlich, 1 = weiblich."""
    sex = np.asarray(sex)
    ss = {lab: np.where(sex == 0, float(m), float(f)) for lab, (m, f) in QTC_SS.items()}
    return ss, dict(QTC_NEUTRAL)


def teilmengen_p0(sex, clbbb):
    return {"alle": np.ones(len(sex), bool), "CLBBB": np.asarray(clbbb, bool), "ohne_CLBBB": ~np.asarray(clbbb, bool),
            "maennlich": np.asarray(sex) == 0, "weiblich": np.asarray(sex) == 1}


def baue_einheiten_p0(D, sex, clbbb):
    """QRS (alle Teilmengen) und QTc (alle, maennlich, weiblich) fuer alle 10 Paare. D: Werkzeug -> dict(qrs, qtc_std, qtc_einh)."""
    K = len(sex)
    w = np.ones(K)
    tm = teilmengen_p0(sex, clbbb)
    ss, neutral = schwellen_qtc(sex)
    qtc_schw = dict(ss)
    qtc_schw.update(neutral)
    einheiten = []
    for i in range(len(WERKZEUGE)):
        for j in range(i + 1, len(WERKZEUGE)):
            ta, tb = S.orientiere(WERKZEUGE[i], WERKZEUGE[j])
            for tmn, mask in tm.items():
                einheiten.append(Einheit("P0|qrs|%s|%s|%s" % (ta, tb, tmn), D[ta]["qrs"], D[tb]["qrs"], w, mask,
                                         {"t%d" % t: t for t in QRS_SCHWELLEN}, klassen=True,
                                         meta=dict(scope="P0", messung="qrs", kandidat=ta, komparator=tb, teilmenge=tmn, gewichtet=False)))
            for messung in ("qtc_std", "qtc_einh"):
                for tmn in ("alle", "maennlich", "weiblich"):
                    einheiten.append(Einheit("P0|%s|%s|%s|%s" % (messung, ta, tb, tmn), D[ta][messung], D[tb][messung], w, tm[tmn], qtc_schw,
                                             meta=dict(scope="P0", messung=messung, kandidat=ta, komparator=tb, teilmenge=tmn, gewichtet=False)))
    return einheiten


def baue_einheiten_sg(D, ix_sg, sex_sg, w_sg, nkw):
    einheiten = []
    tm = {"alle": np.ones(len(sex_sg), bool), "maennlich": np.asarray(sex_sg) == 0, "weiblich": np.asarray(sex_sg) == 1}
    for g, v in nkw.items():
        for t in WERKZEUGE:
            ta, tb = S.orientiere("NeuroKit2", t)
            a = v if ta == "NeuroKit2" else D[ta]["qrs"][ix_sg]
            b = v if tb == "NeuroKit2" else D[tb]["qrs"][ix_sg]
            for tmn, mask in tm.items():
                einheiten.append(Einheit("SG|%s|%s|%s|%s" % (g, ta, tb, tmn), a, b, w_sg, mask, {"t%d" % t_: t_ for t_ in QRS_SCHWELLEN},
                                         klassen=True, meta=dict(scope="S_G", messung=g, kandidat=ta, komparator=tb, teilmenge=tmn, gewichtet=True)))
    return einheiten


# ---------------------------------------------------------------------------------------------------------------
# 3. Tabellen
# ---------------------------------------------------------------------------------------------------------------
def _ki(bs, key):
    lo, hi = ki95(bs[key])
    return lo, hi


def tab_kreuz(einheiten, pk, bs, zl):
    """(a) und (b): Kreuztabelle (lang) und Klassenwechsel je Paar und Teilmenge."""
    kreuz, wechsel = [], []
    for e in einheiten:
        if not e.klassen:
            continue
        m = e.meta
        b, p, z = bs[e.name], pk[e.name], zl[e.name]
        for i in range(4):
            for j in range(4):
                k = "z|%d|%d" % (i, j)
                lo, hi = _ki(b, k)
                kreuz.append(dict(scope=m["scope"], messung=m["messung"], teilmenge=m["teilmenge"], kandidat=m["kandidat"], komparator=m["komparator"],
                                  klasse_kandidat=KLASSEN[i], klasse_komparator=KLASSEN[j], n_zelle=z[k], n_gemeinsam=z["n"],
                                  anteil_pct=100 * p[k], ki_u=100 * lo, ki_o=100 * hi, gewichtet=m["gewichtet"]))
        r = dict(scope=m["scope"], messung=m["messung"], teilmenge=m["teilmenge"], kandidat=m["kandidat"], komparator=m["komparator"],
                 n_gemeinsam=z["n"], gewichtet=m["gewichtet"])
        for k, nm in (("klassenwechsel", "klassenwechsel"), ("wechsel_kandidat_hoeher", "wechsel_kandidat_hoeher"),
                      ("wechsel_kandidat_niedriger", "wechsel_kandidat_niedriger")):
            lo, hi = _ki(b, k)
            r[nm + "_pct"], r[nm + "_ki_u"], r[nm + "_ki_o"] = 100 * p[k], 100 * lo, 100 * hi
        for t in QRS_SCHWELLEN:
            k = "disc|t%d" % t
            lo, hi = _ki(b, k)
            r["D%d_pct" % t], r["D%d_ki_u" % t], r["D%d_ki_o" % t] = 100 * p[k], 100 * lo, 100 * hi
            r["n_disc_%d" % t] = z[k]
        wechsel.append(r)
    return pd.DataFrame(kreuz), pd.DataFrame(wechsel)


def tab_uebereinstimmung(einheiten, pk, bs, zl):
    """(c): Bland-Altman, ICC(2,1), Kappa an 120 ms je Paar und Teilmenge (QRS)."""
    rows = []
    for e in einheiten:
        m = e.meta
        if not e.klassen:
            continue
        b, p, z = bs[e.name], pk[e.name], zl[e.name]
        r = dict(scope=m["scope"], messung=m["messung"], teilmenge=m["teilmenge"], kandidat=m["kandidat"], komparator=m["komparator"],
                 n_gemeinsam=z["n"], gewichtet=m["gewichtet"])
        for k, nm in (("bias", "bias_ms"), ("sd_diff", "sd_diff_ms"), ("loa_u", "loa_untere_ms"), ("loa_o", "loa_obere_ms"), ("icc21", "icc21"),
                      ("kappa|t120", "kappa_120"), ("po|t120", "uebereinstimmung_120"), ("pabak|t120", "pabak_120"), ("disc|t120", "diskordanz_120")):
            lo, hi = _ki(b, k)
            r[nm], r[nm + "_ki_u"], r[nm + "_ki_o"] = p[k], lo, hi
        r["prev120_kandidat"], r["prev120_komparator"] = p["prev_a|t120"], p["prev_b|t120"]
        r["n_nur_kandidat_120"], r["n_nur_komparator_120"] = z["nur_a|t120"], z["nur_b|t120"]
        rows.append(r)
    return pd.DataFrame(rows)


def tab_qtc(einheiten, pk, bs, zl):
    """(d): QTc-Diskordanz je Paar, Messung (Standardformel / einheitlich Fridericia), Teilmenge und Schwellenstufe."""
    rows = []
    for e in einheiten:
        m = e.meta
        if m["messung"] not in ("qtc_std", "qtc_einh"):
            continue
        b, p, z = bs[e.name], pk[e.name], zl[e.name]
        for lab in e.schwellen:
            schema = "geschlechtsspezifisch" if lab.startswith("ss") else "geschlechtsneutral"
            r = dict(scope=m["scope"], messung=m["messung"], teilmenge=m["teilmenge"], kandidat=m["kandidat"], komparator=m["komparator"],
                     schema=schema, stufe=lab, n_gemeinsam=z["n"])
            for k, nm in (("prev_a", "anteil_ge_kandidat"), ("prev_b", "anteil_ge_komparator"), ("disc", "diskordanz"),
                          ("nur_a", "anteil_nur_kandidat"), ("nur_b", "anteil_nur_komparator")):
                lo, hi = _ki(b, "%s|%s" % (k, lab))
                r[nm + "_pct"], r[nm + "_ki_u"], r[nm + "_ki_o"] = 100 * p["%s|%s" % (k, lab)], 100 * lo, 100 * hi
            for k in ("pa", "pb", "disc", "nur_a", "nur_b"):
                r["n_" + k] = z["%s|%s" % (k, lab)]
            for k, nm in (("bias", "bias_ms"), ("loa_u", "loa_untere_ms"), ("loa_o", "loa_obere_ms"), ("icc21", "icc21")):
                lo, hi = _ki(b, k)
                r[nm], r[nm + "_ki_u"], r[nm + "_ki_o"] = p[k], lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


def tab_geschlecht(einheiten, pk, bs, zl):
    """(e): Diskordanz nach Geschlecht und Differenz weiblich minus maennlich (Ziehungsdifferenz)."""
    by = {e.name: e for e in einheiten}
    rows = []
    for e in einheiten:
        m = e.meta
        if m["teilmenge"] != "maennlich":
            continue
        nm_w = e.name[: -len("maennlich")] + "weiblich"
        ew = by.get(nm_w)
        if ew is None:
            continue
        for lab in e.schwellen:
            k = "disc|%s" % lab
            dm, dw = pk[e.name][k], pk[nm_w][k]
            diff = bs[nm_w][k] - bs[e.name][k]
            lo_m, hi_m = _ki(bs[e.name], k)
            lo_w, hi_w = _ki(bs[nm_w], k)
            lo_d, hi_d = ki95(diff)
            rows.append(dict(scope=m["scope"], messung=m["messung"], kandidat=m["kandidat"], komparator=m["komparator"], stufe=lab,
                             n_maennlich=zl[e.name]["n"], n_weiblich=zl[nm_w]["n"],
                             diskordanz_maennlich_pct=100 * dm, dm_ki_u=100 * lo_m, dm_ki_o=100 * hi_m,
                             diskordanz_weiblich_pct=100 * dw, dw_ki_u=100 * lo_w, dw_ki_o=100 * hi_w,
                             differenz_w_minus_m_pp=100 * (dw - dm), diff_ki_u=100 * lo_d, diff_ki_o=100 * hi_d,
                             n_disc_maennlich=zl[e.name]["disc|" + lab], n_disc_weiblich=zl[nm_w]["disc|" + lab], gewichtet=m["gewichtet"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# 4. Laden (zusaetzlich zu S.lade_alles)
# ---------------------------------------------------------------------------------------------------------------
def lade_meta_p0(ids):
    d = pd.read_csv(S.PFADE["meta"], usecols=["ecg_id", "sex", "scp_codes"]).sort_values("ecg_id").set_index("ecg_id").reindex(ids)
    sex = d["sex"].to_numpy(float)
    wk = d["scp_codes"].map(ast.literal_eval)
    clbbb_schluessel = wk.map(lambda x: "CLBBB" in x).to_numpy(bool)
    clbbb_p50 = wk.map(lambda x: x.get("CLBBB", 0) >= 50).to_numpy(bool)
    return sex, clbbb_schluessel, clbbb_p50


def lade_std_qtc(ids, D):
    """Standard-QTc je Werkzeug (D[t]['qtc_std']) und Kontrollspalten (Bazett, Hodge)."""
    P = S.PFADE
    ctrl = {}
    for tool, f, spalten in (("12SL", "12sl_features.csv", ["QT_IntCorr_Global", "QT_IntBazett_Global"]),
                             ("Uni-G", "unig_features.csv", ["QT_IntCorr_Global", "QT_IntHodge_Global"]),
                             ("ECGDeli", "ecgdeli_features.csv", ["QT_IntFramingham_Global"])):
        df = pd.read_csv(P["plus"] / f, usecols=["ecg_id"] + spalten)
        df = df[~df.ecg_id.duplicated(keep="first")].set_index("ecg_id").reindex(ids)
        D[tool]["qtc_std"] = df[spalten[0]].to_numpy(float)
        ctrl[tool] = df
    D["LoopLens"]["qtc_std"] = D["LoopLens"]["qtc_nativ"]
    D["BRAVEHEART"]["qtc_std"] = D["BRAVEHEART"]["qtc_nativ"]
    return ctrl


# ---------------------------------------------------------------------------------------------------------------
# 5. Lauf
# ---------------------------------------------------------------------------------------------------------------
def kontrollen_vor_bootstrap(einheiten_p0, ctrl, sex, clbbb, clbbb50, ids):
    by = {e.name: e for e in einheiten_p0}
    for (ta, tb), (n_soll, bias_soll, d120_soll) in S.Z1_ERWARTET.items():
        e = by["P0|qrs|%s|%s|alle" % (ta, tb)]
        pk = e.punkt()
        S.kontrolle("K1 Nachbau AW-228 %s gegen %s: n, Bias, D120" % (ta, tb), "%d, %.2f, %.1f %%" % (n_soll, bias_soll, d120_soll),
                    "%d, %.2f, %.2f %%" % (e.n, pk["bias"], 100 * pk["disc|t120"]),
                    e.n == n_soll and abs(pk["bias"] - bias_soll) <= 0.0101 and abs(100 * pk["disc|t120"] - d120_soll) <= 0.051)
    # ICC: Momentenformel gegen direkte Varianzanalyse, alle QRS-Paare (alle)
    abw = 0.0
    for nm, e in by.items():
        if e.klassen and nm.endswith("|alle"):
            abw = max(abw, abs(e.punkt()["icc21"] - icc21_direkt(e.a, e.b)))
    S.kontrolle("K2 ICC(2,1) Momentenformel gegen Varianzanalyse (alle QRS-Paare)", "max Abweichung <= 1e-9", "%.2e" % abw, abw <= 1e-9)
    # Kreuztabelle summiert auf n
    ok = all(abs(sum(e.punkt()["z|%d|%d" % (i, j)] for i in range(4) for j in range(4)) - 1.0) < 1e-9 for e in einheiten_p0 if e.klassen and e.n > 0)
    S.kontrolle("K3 Kreuztabellenzellen summieren auf 100 %", "ja", "ja" if ok else "nein", ok)
    # Standard-QTc-Spalten
    for tool, a, b in (("12SL", "QT_IntCorr_Global", "QT_IntBazett_Global"), ("Uni-G", "QT_IntCorr_Global", "QT_IntHodge_Global")):
        df = ctrl[tool]
        x, y = df[a].to_numpy(float), df[b].to_numpy(float)
        gleich = bool(np.all((np.isnan(x) & np.isnan(y)) | (x == y)))
        S.kontrolle("K4 %s %s gleich %s (Standardformel laut Tabelle 1)" % (tool, a, b), "alle gleich", "alle gleich" if gleich else "Abweichung", gleich)
    # Geschlecht und CLBBB
    sexwerte = sorted(set(np.unique(sex[np.isfinite(sex)]).tolist()))
    S.kontrolle("K5 sex nur 0/1, vollstaendig", "[0.0, 1.0], 0 fehlend", "%s, %d fehlend" % (sexwerte, int(np.isnan(sex).sum())),
                sexwerte == [0.0, 1.0] and not np.isnan(sex).any())
    S.kontrolle("K6 CLBBB-Teilmenge: Schluessel (V2) und Wahrscheinlichkeit >= 50 (nur Information, bricht nicht ab)", "Anzahlen ausweisen",
                "%d gegen %d, Schnittmenge %d" % (clbbb.sum(), clbbb50.sum(), (clbbb & clbbb50).sum()), bool(clbbb50.sum() <= clbbb.sum()),
                hinweis="Wahrscheinlichkeit >= 50 ist Teilmenge des Schluessels; Unterschied wird im Bericht ausgewiesen")
    S.kontrolle("K7 CLBBB-Teilmenge nicht leer", ">= 100", str(int(clbbb.sum())), int(clbbb.sum()) >= 100)


def schreibe(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    KOPIE.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, encoding="utf-8", float_format="%.6g")
    df.to_csv(KOPIE / name, index=False, encoding="utf-8", float_format="%.6g")


def lauf():
    P = S.PFADE
    OUT.mkdir(parents=True, exist_ok=True)
    S.STATE["log"] = str(OUT / "aw234_auswertung.log")
    t0 = time.time()
    me = Path(__file__)
    log("Start aw234_auswertung Version %s, Skript-SHA-256 %s, importiert schwellen_aw228 %s" % (VERSION, S.sha256_datei(me), S_HASH))
    ok = S.hashpruefung(S.gelesene_dateien())
    pd.DataFrame(S.STATE["pruef"]).to_csv(OUT / "00_hashpruefung.csv", index=False, encoding="utf-8")
    log("Hashpruefung: %d Eintraege, alle stimmen: %s" % (len(S.STATE["pruef"]), ok))
    if not ok:
        log("ABBRUCH: Hashpruefung nicht bestanden")
        return 2
    ids, D = S.lade_alles()
    sex, clbbb, clbbb50 = lade_meta_p0(ids)
    ctrl = lade_std_qtc(ids, D)
    einheiten = baue_einheiten_p0(D, sex, clbbb)
    kontrollen_vor_bootstrap(einheiten, ctrl, sex, clbbb, clbbb50, ids)
    sg, st, w_sg, nkw = S.lade_sg_nk(ids)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    ix_sg = pos[sg].to_numpy()
    eh_sg = baue_einheiten_sg(D, ix_sg, sex[ix_sg], w_sg, nkw)
    if not all(k["bestanden"] for k in S.STATE["kontrollen"]):
        pd.DataFrame(S.STATE["kontrollen"]).to_csv(OUT / "01_kontrollen.csv", index=False, encoding="utf-8")
        log("ABBRUCH: eine Kontrolle ist nicht bestanden (vor der Bootstrap-Rechnung)")
        return 3
    # Verfuegbarkeit
    av = []
    for t in WERKZEUGE:
        av.append(dict(werkzeug=t, n_qrs=int(np.isfinite(D[t]["qrs"]).sum()), n_qtc_standard=int(np.isfinite(D[t]["qtc_std"]).sum()),
                       n_qtc_fridericia_einheitlich=int(np.isfinite(D[t]["qtc_einh"]).sum()), von=len(ids)))
    schreibe(pd.DataFrame(av), "T0_verfuegbarkeit.csv")
    # Randverteilungen (Einordnung)
    rv = []
    for t in WERKZEUGE:
        for key in ("qrs", "qtc_std", "qtc_einh"):
            x = D[t][key]
            x = x[np.isfinite(x)]
            rv.append(dict(werkzeug=t, messung=key, n=len(x), mittel=float(x.mean()), median=float(np.median(x)), p05=float(np.percentile(x, 5)),
                           p95=float(np.percentile(x, 95)), minimum=float(x.min()), maximum=float(x.max())))
    schreibe(pd.DataFrame(rv), "T0_randverteilung.csv")
    # Teilmengengroessen
    tm = teilmengen_p0(sex, clbbb)
    schreibe(pd.DataFrame([dict(teilmenge=k, n_personen=int(v.sum())) for k, v in tm.items()] +
                          [dict(teilmenge="CLBBB_Wahrscheinlichkeit_ge_50", n_personen=int(clbbb50.sum()))]), "T0_teilmengen.csv")
    log("Bootstrap P0: %d Einheiten, %d Ziehungen" % (len(einheiten), B_ZIEHUNGEN))
    bs = boot_lauf(einheiten, S.Ziehungen(len(ids), SEED))
    log("Bootstrap S_G: %d Einheiten" % len(eh_sg))
    bs_sg = boot_lauf(eh_sg, S.Ziehungen(len(sg), SEED_SG, strata=st))
    alle = einheiten + eh_sg
    bs.update(bs_sg)
    pk = {e.name: e.punkt() for e in alle}
    zl = {e.name: e.zaehlungen() for e in alle}
    kreuz, wechsel = tab_kreuz(alle, pk, bs, zl)
    schreibe(kreuz, "T1_kreuztabelle_qrs_klassen.csv")
    schreibe(wechsel, "T2_klassenwechsel_und_diskordanz.csv")
    schreibe(tab_uebereinstimmung(alle, pk, bs, zl), "T3_bland_altman_icc_kappa.csv")
    schreibe(tab_qtc(einheiten, pk, bs, zl), "T4_qtc_schwellen.csv")
    schreibe(tab_geschlecht(alle, pk, bs, zl), "T5_diskordanz_nach_geschlecht.csv")
    pd.DataFrame(S.STATE["kontrollen"]).to_csv(OUT / "01_kontrollen.csv", index=False, encoding="utf-8")
    pd.DataFrame(S.STATE["kontrollen"]).to_csv(KOPIE / "01_kontrollen.csv", index=False, encoding="utf-8")
    (KOPIE / "00_hashpruefung.csv").write_bytes((OUT / "00_hashpruefung.csv").read_bytes())
    info = dict(version=VERSION, skript_sha256=S.sha256_datei(me), schwellen_aw228_sha256=S_HASH, seed=SEED, seed_sg=SEED_SG, ziehungen=B_ZIEHUNGEN,
                chunk=CHUNK, start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), ende=time.strftime("%Y-%m-%d %H:%M:%S"),
                python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, hashpruefung_eintraege=len(S.STATE["pruef"]),
                kontrollen=len(S.STATE["kontrollen"]), kontrollen_nicht_bestanden=sum(1 for k in S.STATE["kontrollen"] if not k["bestanden"]),
                qtc_schwellenquelle=QTC_QUELLE, clbbb_definition="Schluessel CLBBB in scp_codes (V2 4.2, bench_auswertung_v11.py); Empfindlichkeit: Wahrscheinlichkeit >= 50",
                n_clbbb_schluessel=int(clbbb.sum()), n_clbbb_p50=int(clbbb50.sum()))
    (OUT / "LAUF.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    (KOPIE / "LAUF.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log("Ende nach %.0f s" % (time.time() - t0))
    return 0


def haupt(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", choices=["hash", "lauf"], required=True)
    a = ap.parse_args(argv)
    if a.modus == "hash":
        ok = S.hashpruefung(S.gelesene_dateien())
        for z in S.STATE["pruef"]:
            print(("OK   " if z["ok"] else "FEHL ") + str(z["datei"]))
        print("alle stimmen:", ok, "Eintraege:", len(S.STATE["pruef"]))
        return 0 if ok else 2
    return lauf()


if __name__ == "__main__":
    sys.exit(haupt())
